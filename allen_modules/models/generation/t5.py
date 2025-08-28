from os import PathLike
from typing import Optional, Dict, Any, Union, List, Tuple
import time
import numpy as np
import nltk
import torch
# Set the maximal number of CPU cores
torch.set_num_threads(4)

from allennlp.common.lazy import Lazy
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.models.model import Model
# from allennlp.modules.transformer.t5 import T5 as T5Module
from allennlp.modules.transformer.t5 import T5Output, IntT, BoolT
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.checkpoint import CheckpointWrapper
from allennlp.training.metrics import ROUGE, BLEU, Auc, F1Measure, F1MultiLabelMeasure, Metric, EvalbBracketingScorer

from allen_modules.training.metrics.exact_match import ExactMatchAcc
from allen_modules.training.metrics.exact_match_tokens import TokenMatchAcc
from allen_modules.training.metrics.amr import AMRMetrics
from allen_modules.training.metrics.epoch import EpochsPassed
from allen_modules.training.metrics.expected_calibration_error import ExpectedCalibrationError, TokenExpectedCalibrationError
from allen_modules.training.postprocess.postprocessor import Postprocessor
from allen_modules.training.postprocess.simple import SimplePostprocessor
from allen_modules.modules.transformer.t5 import T5 as T5Module
from allen_modules.evaluation.logger.T5_features import T5FeaturesLogger
from allen_modules.evaluation.logger.logger import Logger

@Model.register("modified_t5")
class T5(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model_name: str,
        beam_search: Lazy[BeamSearch] = Lazy(BeamSearch, beam_size=3, max_steps=50),
        checkpoint_wrapper: Optional[CheckpointWrapper] = None,
        weights_path: Optional[Union[str, PathLike]] = None,
        postprocessor: Postprocessor = None,
        val_ece: bool = False,
        label_smoothing: float = None,
        output_attentions: bool = False,
        metrics: Optional[List[Metric]] = None,
        logger: Logger = None,
        # test_dropout: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._model_name = model_name
        # We only instantiate this when we need it.
        self._tokenizer: Optional[PretrainedTransformerTokenizer] = None
        # print("#####DEBUG: Dropout: {}".format(test_dropout))
        self.t5 = T5Module.from_pretrained_module(
            model_name,
            beam_search=beam_search,
            ddp_accelerator=self.ddp_accelerator,
            checkpoint_wrapper=checkpoint_wrapper,
            weights_path=weights_path,
            label_smoothing=label_smoothing,
            output_attentions=output_attentions,
            # test_dropout=test_dropout,
        )
        exclude_indices = {
            self.t5.pad_token_id,
            self.t5.decoder_start_token_id,
            self.t5.eos_token_id,
        }
        self.postprocessor = postprocessor

        self._metrics = metrics

        # For ece computation
        self.val_ece = val_ece
        self.output_attentions = output_attentions
        self.logger = logger
        if val_ece:
            self._ece = ExpectedCalibrationError(interval=0.2)
            self._metrics.append(self._ece)
            self._token_ece = TokenExpectedCalibrationError(interval=0.2)
            self._metrics.append(self._token_ece)
    def _post_load_state_dict(
        self, missing_keys: List[str], unexpected_keys: List[str]
    ) -> Tuple[List[str], List[str]]:
        missing_keys_to_ignore = [
            "t5.encoder.token_embeddings.weight",
            "t5.decoder.token_embeddings.weight",
        ]
        if self.t5._tie_word_embeddings:
            missing_keys_to_ignore.append("t5.lm_head.weight")
        for key in missing_keys_to_ignore:
            if key in missing_keys:
                missing_keys.remove(key)
        return missing_keys, unexpected_keys

    @property
    def tokenizer(self) -> PretrainedTransformerTokenizer:
        if self._tokenizer is None:
            self._tokenizer = PretrainedTransformerTokenizer(self._model_name)
        return self._tokenizer

    def forward(  # type: ignore
        self, source_tokens: TextFieldTensors,
            target_tokens: Optional[TextFieldTensors] = None,
            metadata: Dict = None
    ) -> Dict[str, torch.Tensor]:
        """
        Performs the forward step of T5.

        # Parameters

        source_tokens : `TextFieldTensors`, required
            The source tokens for the encoder. We assume they are stored under the `tokens` key/namespace.

        target_tokens : `TextFieldTensors`, optional (default = `None`)
            The target tokens for the decoder. We assume they are also stored under the `tokens` key/namespace.
            If no target tokens are given during training / validation, the source tokens are shifted
            to the right by 1.

        # Returns

        `Dict[str, torch.Tensor]`
            Contains the `loss` when `target_tokens` is provided.
            And during prediction, includes `predictions` and `predicted_log_probs` from beam search.

        """
        time0 = time.time()
        input_ids, attention_mask = (
            source_tokens["tokens"]["token_ids"],
            source_tokens["tokens"]["mask"],
        )
        labels: Optional[IntT] = None
        decoder_attention_mask: Optional[BoolT] = None
        if target_tokens is not None:
            labels, decoder_attention_mask = (
                target_tokens["tokens"]["token_ids"],  # type: ignore[assignment]
                target_tokens["tokens"]["mask"],  # type: ignore[assignment]
            )
        elif self.training:
            raise ValueError("'target_tokens' required during training")

        output: T5Output = self.t5(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        output_dict: Dict[str, torch.Tensor] = {}

        # print("forward time: {}".format(time.time() - time0))
        # time0 = time.time()

        if self.training:
            assert output.loss is not None
            output_dict["loss"] = output.loss
        else:
            # Shape: (batch_size, beam_size, num_tokens)
            assert output.predictions is not None
            # Shape: (batch_size, beam_size)
            assert output.predicted_log_probs is not None
            # Shape: (batch_size, num_tokens)
            output_dict["predictions"] = output.predictions[:, 0, :]
            # Shape: (batch_size, )
            output_dict["predicted_log_probs"] = output.predicted_log_probs[:, 0]
            output_dict["beam_predicted_log_probs"] = [output.predicted_log_probs[:, i] for i in range(output.predicted_log_probs.size(1))]

            output_dict = self.make_output_human_readable(output_dict, beam_predictions=output.predictions)
            # print("make_output_human_readable time: {}".format(time.time() - time0))

            # Whether use minimal token probability as the sequence confidence
            use_min_token_conf = False
            if use_min_token_conf:
                output_dict["predicted_probs"] = self.maintain_min_seq_confidence(input_ids, attention_mask,
                                                 predictions=output_dict["predictions"]).tolist()
                output_dict["beam_predicted_probs"] = [self.maintain_min_seq_confidence(input_ids, attention_mask,
                                                 predictions=p).tolist() for p in output_dict["beam_predicted_log_probs"]]
            else:
                output_dict["predicted_probs"] = torch.exp(output.predicted_log_probs[:, 0]).tolist()
                output_dict["beam_predicted_probs"] = [torch.exp(p).tolist() for p in output_dict["beam_predicted_log_probs"]]

            # Save inputs for Bertviz script if applied
            if self.output_attentions:
                # self.prepare_viz(encoder_input_ids=input_ids,
                #                  decoder_input_ids=output.predictions[:, 0, :],
                #                  decoder_attentions=output.decoder_attentions,
                #                  cross_attentions=output.cross_attentions,
                #                  metadata=metadata)
                output_dict["cross_attentions"] = output.cross_attentions
                pass
            # print(output_dict["predicted_probs"])

            if labels is not None:

                self.calc_metrics(output_dict, labels, metadata)

                # Save loss of each instance for ece computation and output
                assert output.loss is not None
                output_dict["loss"] = output.loss.mean()
                batch_size, seq_len = labels.size()
                loss_per_ins = output.loss.view(batch_size, seq_len)
                loss_per_ins = torch.sum(loss_per_ins, dim=1, keepdim=False)
                output_dict["loss_per_ins"] = loss_per_ins.tolist()
                output_dict["gold_probs"] = torch.exp(-1 * loss_per_ins).tolist()

                # Compute seq-ece and token-ece if applied
                if self.val_ece:
                    self._ece(output_dict["predicted_text"], metadata, output_dict["predicted_probs"])
                    output_dict = self.maintain_sequence_confidence(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        decoder_attention_mask=decoder_attention_mask,
                        output_dict=output_dict,
                        metadata=metadata
                    )
                    self._token_ece(output_dict["token_pred_confidence"],
                                    output_dict["token_pred_acc"],
                                    metadata
                                    )

                if self.logger:
                    output_dict = self.logger(input_ids, output_dict, metadata)

        # print("other time: {}".format(time.time() - time0))

        return output_dict

    def maintain_sequence_confidence(self, input_ids, attention_mask,
                                labels, decoder_attention_mask,
                                output_dict,
                                metadata=None):
        pred_confidence, pred_indices = self.t5.forward_sequence(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        pred_acc = (pred_indices == labels).type(torch.float32)
        pred_confidence = pred_confidence.masked_fill(~decoder_attention_mask, -1)
        pred_acc = pred_acc.masked_fill(~decoder_attention_mask, -1)

        output_dict["token_pred_confidence"] = pred_confidence.tolist()
        output_dict["token_pred_acc"] = pred_acc.tolist()


        return output_dict

    def maintain_min_seq_confidence(self, input_ids, attention_mask, predictions):
        # print(predictions[:5, :].tolist())
        # print(labels[:5, :].tolist())
        # print(decoder_attention_mask[:5, :].tolist())
        # raise NotImplementedError

        # create a mask tensor to mask all 1 values
        mask_tensor = (predictions != 1)
        # print(mask_tensor[8, :].tolist())
        # iterate over each row in the mask tensor
        for i in range(mask_tensor.size(0)):
            row = mask_tensor[i]
            first_false_idx = (row == False).nonzero(as_tuple=True)[0]  # get the index of the first False value in the row
            if first_false_idx.nelement() > 0:
                mask_tensor[i, first_false_idx[0]] = True
        pred_confidence, pred_indices = self.t5.forward_sequence(
            input_ids,
            attention_mask=attention_mask,
            labels=predictions,
            decoder_attention_mask=mask_tensor,
        )
        pred_confidence = pred_confidence.masked_fill(~mask_tensor, 1)
        min_seq_confidence, _ = torch.min(pred_confidence, 1)
        # prod_confidence = torch.prod(pred_confidence, dim=1)
        # print(prod_confidence)
        # print(mask_tensor[8, :].tolist())
        # print(predictions[8, :].tolist())
        # print(min_seq_confidence.tolist())
        return min_seq_confidence

    def prepare_viz(self, encoder_input_ids, decoder_input_ids, cross_attentions, decoder_attentions, metadata):
        assert len(metadata) < 20
        # print(len(cross_attentions))
        # print(cross_attentions[0].size())
        # print(cross_attentions[1].size())
        # print(cross_attentions[2].size())

        for i in range(len(metadata)):
            if "gen_type" not in metadata[0]:
                gen_type = "test_{}".format(i)
            else:
                gen_type = metadata[i]["gen_type"]
            eg_dict = {"encoder_input_ids": encoder_input_ids[i],
                       "decoder_input_ids": decoder_input_ids[i],
                       "decoder_attentions": tuple(decoder_attentions[k][i].unsqueeze(0) for k in range(12)),
                       "cross_attention": tuple(cross_attentions[k][i].unsqueeze(0) for k in range(12))}
            torch.save(eg_dict, "model_archives/cogs/confidence/T5_simp/output/{}.pt".format(gen_type))


    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor], beam_predictions=None) -> Dict[str, Any]:
        # print(output_dict.keys())
        predictions = output_dict["predictions"]
        # print(predictions[0])
        # for i in range(len(predictions)):
        #     for j in range(len(predictions[i])):
        #         print(self.tokenizer.tokenizer.decode(predictions[i][j], skip_special_tokens=self.postprocessor.skip_special_tokens, clean_up_tokenization_spaces=False  # type: ignore[attr-defined]
        #     ))
        #     print(self.tokenizer.tokenizer.decode(
        #     predictions[i], skip_special_tokens=self.postprocessor.skip_special_tokens, clean_up_tokenization_spaces=False  # type: ignore[attr-defined]
        # ))
        #     raise NotImplementedError
        predicted_texts = self.tokenizer.tokenizer.batch_decode(
            predictions, skip_special_tokens=self.postprocessor.skip_special_tokens if self.postprocessor is not None else True, clean_up_tokenization_spaces=False  # type: ignore[attr-defined]
        )

        if self.postprocessor is not None:
            output_dict["predicted_text"] = self.postprocessor(predicted_texts)
        else:
            output_dict["predicted_text"] = predicted_texts

        if beam_predictions is not None:
            beam_predicted_texts = []
            # Shape (batch_size, beam_size, num_tokens)
            for i in range(beam_predictions.size(1)):
               predictions_beam_i = beam_predictions[:, i, :]
               predicted_texts_beam_i = self.tokenizer.tokenizer.batch_decode(
                    predictions_beam_i, skip_special_tokens=self.postprocessor.skip_special_tokens if self.postprocessor is not None else True, clean_up_tokenization_spaces=False  # type: ignore[attr-defined]
                  )
               beam_predicted_texts.append(predicted_texts_beam_i)

            if self.postprocessor is not None:
                output_dict["beam_predicted_text"] = [self.postprocessor(beam_predicted_text) for beam_predicted_text in beam_predicted_texts]
            else:
                output_dict["beam_predicted_text"] = beam_predicted_texts

        return output_dict

    def calc_metrics(self, output_dict: Dict[str, Any], labels: torch.Tensor, metadata: List) -> None:
        """
        Use the predicted sequence and the gold sequence to compute metrics.
        """
        # Compute exact match accuracy as main validation metric
        for metric in self._metrics:
            if isinstance(metric, Auc):
                probs = output_dict["predicted_probs"]
                predicted_texts = output_dict["predicted_text"]
                label_probs = torch.Tensor(
                    [probs[i] if predicted_texts[i].lower() == "true" else 1 - probs[i] for i in range(len(probs))])
                gold_labels = torch.Tensor(
                    [1 if metadata[i]["target_text"].lower() == "true" else 0 for i in range(len(metadata))])
                metric(label_probs, gold_labels)
            if isinstance(metric, F1Measure):
                probs = output_dict["predicted_probs"]
                predicted_texts = output_dict["predicted_text"]
                label_probs = torch.Tensor(
                    [[1 - probs[i], probs[i]] if predicted_texts[i].lower() == "true" else [probs[i], 1 - probs[i]] for i in
                     range(len(probs))])
                gold_labels = torch.Tensor(
                    [1 if metadata[i]["target_text"].lower() == "true" else 0 for i in range(len(metadata))])
                metric(label_probs, gold_labels)
            if isinstance(metric, ROUGE):
                metric(output_dict["predictions"], labels)
            if isinstance(metric, BLEU):
                metric(output_dict["predictions"], labels)
            if isinstance(metric, ExactMatchAcc):
                # print("#####DEBUG: Exact match#####")
                metric(output_dict["predicted_text"], metadata)
            # if isinstance(metric, GeoExecuteAcc):
            #     metric(output_dict["predicted_text"], metadata)
            if isinstance(metric, EpochsPassed):
                metric()
            # if isinstance(metric, LocalstructureDicrimMatch):
            #     metric(output_dict["predicted_text"], metadata)
            if isinstance(metric, AMRMetrics):
                metric(output_dict["predicted_text"], metadata)
            if isinstance(metric, EvalbBracketingScorer):
                predicted_trees = []
                gold_trees = []
                for i in range(len(output_dict["predicted_text"])):
                    readable = True
                    try:
                        predicted_trees.append(nltk.Tree.fromstring(output_dict["predicted_text"][i]))  # type: ignore
                    except:
                        string = output_dict["predicted_text"][i]
                        print("#####WARNING: Ignore Predicted text: {}#####".format(string))
                        readable = False
                    if readable:
                        gold_trees.append(nltk.Tree.fromstring(metadata[i]["target_text"]))  # type: ignore
                try:
                    metric(predicted_trees, gold_trees)
                except:
                    print("#####WARNING: Ignore this iteration #####")
                    # raise NotImplementedError
            if isinstance(metric, TokenMatchAcc):
                metric(output_dict["predicted_text"], metadata)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not self.training:
            for metric in self._metrics:
                if isinstance(metric, Auc):
                    metrics["Auc"] = metric.get_metric(reset=reset)
                elif isinstance(metric, F1Measure):
                    label = "True" if metric._positive_label == 1 else "False"
                    metrics.update({f"{label}_{k}": v for k, v in metric.get_metric(reset=reset).items()})
                else:
                    metrics.update(metric.get_metric(reset=reset))
        return metrics

    @classmethod
    def from_archive(cls, archive_file: str, vocab: Vocabulary = None, weights_file:str = None,
                     beam_search:Dict=None) -> "Model":
        """
        Loads a model from an archive file.  This basically just calls
        `return archival.load_archive(archive_file).model`.  It exists as a method here for
        convenience, and so that we can register it for easy use for fine tuning an existing model
        from a configs file.

        If `vocab` is given, we will extend the loaded model's vocabulary using the passed vocab
        object (including calling `extend_embedder_vocab`, which extends embedding layers).
        """
        from allennlp.models.archival import load_archive  # here to avoid circular imports

        # print("#####DEBUG#####")
        # print("Loading model from archive file at {}".format(archive_file))
        model = load_archive(archive_file, weights_file=weights_file).model
        if vocab:
            model.vocab.extend_from_vocab(vocab)
            model.extend_embedder_vocab()
        if beam_search is not None:
            beam_size = beam_search["beam_size"]
            model.t5.beam_search.beam_size = beam_size
            print("set beam size to {}".format(beam_size))
        # raise NotImplementedError
        return model

    default_predictor = "seq2seq"

Model.register("from_archive_T5_beam", constructor="from_archive")(T5)
