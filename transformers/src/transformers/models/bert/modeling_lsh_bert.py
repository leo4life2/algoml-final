from torch import device
from .modeling_bert import *

logger = logging.get_logger(__name__)

import time

class Timer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Timer, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self):
        self.start_times = []
        self.end_times = []
        self.dot_prods = 0
        self.flops = 0
        
    def record_dot(self, number):
        self.dot_prods += number
    
    def record_flops(self, number):
        self.flops += number

    def start(self):
        self.start_times.append(time.perf_counter())

    def stop(self):
        self.end_times.append(time.perf_counter())

    def average_time(self):
        if len(self.start_times) != len(self.end_times):
            raise ValueError("Number of start and stop times do not match")
        total_time = sum(end - start for start, end in zip(self.start_times, self.end_times))
        return total_time / len(self.start_times) if self.start_times else 0
    
timer = Timer()
    
class BertLSHModel(BertModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertLSHEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
        
def sign_hash_function(vectors, x):
    """Compute the sign hash for a given vector x using vectorized operations."""
    return torch.sign(torch.matmul(vectors, x))

class LSH:
    def __init__(self, bands, table_size, num_hashes, feature_size, device):
        self.bands = bands
        self.table_size = table_size
        self.num_hashes = num_hashes
        self.feature_size = feature_size
        self.device = device
        self.random_vectors = torch.randn(self.num_hashes, self.bands, self.feature_size, device=device)
        # Initialize random coefficients for each hash function
        self.hash_coefficients = torch.randint(0, table_size, (num_hashes, bands), device=device)
        self.bucket_table = [{} for _ in range(self.num_hashes)] # use a dictionary so we don't store buckets with nothing at all
        
    def _uniform_random_hash(self, vector, hash_idx):
        """Apply a uniform random hash function to the vector using vectorized operations."""
        # Vectorized computation of the hash value
        hash_values = (vector > 0) * self.hash_coefficients[hash_idx]
        hash_value = hash_values.sum() % self.table_size
        return hash_value.item()

    def _hash_to_table(self, vectors):
        """Compute the hashes for all vectors using all hash functions."""
        # Reshape random_vectors to combine num_hashes and bands for batch matrix multiplication
        random_vectors_reshaped = self.random_vectors.view(-1, self.feature_size)
        # Perform batch matrix multiplication
        sign_vectors = torch.sign(torch.matmul(random_vectors_reshaped, vectors.T))
        # Reshape sign_vectors back to [num_hashes, num_vectors, bands]
        sign_vectors = sign_vectors.view(self.num_hashes, self.bands, -1).transpose(1, 2)

        # Expand hash_coefficients to match the dimensions of sign_vectors
        expanded_hash_coefficients = self.hash_coefficients.view(self.num_hashes, 1, self.bands).expand(-1, vectors.shape[0], -1)

        # Perform the multiplication operation and sum across bands
        hash_values = ((sign_vectors > 0) * expanded_hash_coefficients).sum(dim=2) % self.table_size

        # Populate the bucket_table
        for hash_f_ix in range(self.num_hashes):
            unique_hashes, inverse_indices, counts = torch.unique(hash_values[hash_f_ix], return_inverse=True, return_counts=True)
            for i, hash_val in enumerate(unique_hashes):
                indices = torch.where(inverse_indices == i)[0]
                if len(indices) > 1:
                    self.bucket_table[hash_f_ix].setdefault(hash_val.item(), set()).update(indices.tolist())


    def do_lsh(self, vectors):
        """Perform LSH on all vectors and return the collision matrix using vectorized operations."""
        num_vectors = vectors.shape[0]
        collision_matrix = torch.zeros((num_vectors, num_vectors), device=self.device)

        self._hash_to_table(vectors)

        # Construct collision_matrix
        for hash_index, bucket_dict in enumerate(self.bucket_table):
            for bucket_key, vector_indices_set in bucket_dict.items():
                vector_indices = torch.tensor(list(vector_indices_set), device=self.device)
                collision_matrix.index_put_((vector_indices[:, None], vector_indices), torch.tensor(1, dtype=collision_matrix.dtype, device=self.device))

        return collision_matrix[:num_vectors // 2, num_vectors // 2:] # only interested in Q vs K indices.
    
class BertLSHSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

        # LSH parameters
        self.bands = config.bands
        self.table_size = config.table_size
        self.num_hashes = config.num_hashes

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)
            
            
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        ### START LSH ATTENTION ###
        # Prepare empty attention scores
        # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], 
        #             record_shapes=True, 
        #             profile_memory=True,
        #             with_flops=True) as prof:
        batch_size, num_heads, seq_len, _ = query_layer.shape
        attention_scores = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=query_layer.device)
        
        for batch in range(batch_size):
            for head in range(num_heads):
                q = query_layer[batch, head, :, :]
                k = key_layer[batch, head, :, :]
                
                # stack vertically for LSH
                stacked = torch.cat((q, k), dim=0)
                lsh = LSH(bands=self.bands, table_size=self.table_size, num_hashes=self.num_hashes, feature_size=stacked.size(1), device=query_layer.device)
                
                collision_matrix = lsh.do_lsh(stacked)  # returns a seq_len x seq_len matrix
                
                # Use boolean indexing to find collided indices
                collided_indices = torch.triu_indices(seq_len, seq_len, offset=0, device=query_layer.device)
                collided_mask = collision_matrix[collided_indices[0], collided_indices[1]] == 1
                
                # Filter the indices where collisions occurred
                q_indices = collided_indices[0][collided_mask]
                k_indices = collided_indices[1][collided_mask]
                
                # timer.record_dot(q_indices.shape[0])
                
                # Compute the dot products for collided pairs
                q_vecs = q[q_indices]
                k_vecs = k[k_indices]
                attn_scores = (q_vecs * k_vecs).sum(dim=1)
                
                # Assign the computed attention scores
                attention_scores[batch, head, q_indices, k_indices] = attn_scores
                attention_scores[batch, head, k_indices, q_indices] = attn_scores  # symmetric assignment
        ### END LSH ATTENTION ###
        # for avg in prof.key_averages():
        #     if avg.key == "aten::mm":
        #         timer.record_flops(avg.flops)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class BertLSHAttention(BertAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        self.self = BertLSHSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()


class BertLSHLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertLSHAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertLSHAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertLSHEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLSHLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# @add_start_docstrings(
#     """
#     Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
#     sentence prediction (classification)` head.
#     """,
#     BERT_START_DOCSTRING,
# )
# class BertLSHForPreTraining(BertPreTrainedModel):
#     _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

#     def __init__(self, config):
#         super().__init__(config)

#         self.bert = BertLSHModel(config)
#         self.cls = BertPreTrainingHeads(config)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_output_embeddings(self):
#         return self.cls.predictions.decoder

#     def set_output_embeddings(self, new_embeddings):
#         self.cls.predictions.decoder = new_embeddings

#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @replace_return_docstrings(output_type=BertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         next_sentence_label: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
#         r"""
#             labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
#                 config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
#                 the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
#             next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#                 Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
#                 pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

#                 - 0 indicates sequence B is a continuation of sequence A,
#                 - 1 indicates sequence B is a random sequence.
#             kwargs (`Dict[str, any]`, optional, defaults to *{}*):
#                 Used to hide legacy arguments that have been deprecated.

#         Returns:

#         Example:

#         ```python
#         >>> from transformers import AutoTokenizer, BertForPreTraining
#         >>> import torch

#         >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#         >>> model = BertForPreTraining.from_pretrained("bert-base-uncased")

#         >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
#         >>> outputs = model(**inputs)

#         >>> prediction_logits = outputs.prediction_logits
#         >>> seq_relationship_logits = outputs.seq_relationship_logits
#         ```
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output, pooled_output = outputs[:2]
#         prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

#         total_loss = None
#         if labels is not None and next_sentence_label is not None:
#             loss_fct = CrossEntropyLoss()
#             masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
#             next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
#             total_loss = masked_lm_loss + next_sentence_loss

#         if not return_dict:
#             output = (prediction_scores, seq_relationship_score) + outputs[2:]
#             return ((total_loss,) + output) if total_loss is not None else output

#         return BertForPreTrainingOutput(
#             loss=total_loss,
#             prediction_logits=prediction_scores,
#             seq_relationship_logits=seq_relationship_score,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# @add_start_docstrings(
#     """Bert Model with a `language modeling` head on top for CLM fine-tuning.""", BERT_START_DOCSTRING
# )
# class BertLSHLMHeadModel(BertPreTrainedModel):
#     _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

#     def __init__(self, config):
#         super().__init__(config)

#         if not config.is_decoder:
#             logger.warning("If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

#         self.bert = BertLSHModel(config, add_pooling_layer=False)
#         self.cls = BertOnlyMLMHead(config)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_output_embeddings(self):
#         return self.cls.predictions.decoder

#     def set_output_embeddings(self, new_embeddings):
#         self.cls.predictions.decoder = new_embeddings

#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=CausalLMOutputWithCrossAttentions,
#         config_class=_CONFIG_FOR_DOC,
#     )
#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         past_key_values: Optional[List[torch.Tensor]] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
#         r"""
#         encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
#             Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
#             the model is configured as a decoder.
#         encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
#             the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
#             `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
#             ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
#         past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
#             Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

#             If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
#             don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
#             `decoder_input_ids` of shape `(batch_size, sequence_length)`.
#         use_cache (`bool`, *optional*):
#             If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
#             `past_key_values`).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         if labels is not None:
#             use_cache = False

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]
#         prediction_scores = self.cls(sequence_output)

#         lm_loss = None
#         if labels is not None:
#             # we are doing next-token prediction; shift prediction scores and input ids by one
#             shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
#             labels = labels[:, 1:].contiguous()
#             loss_fct = CrossEntropyLoss()
#             lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

#         if not return_dict:
#             output = (prediction_scores,) + outputs[2:]
#             return ((lm_loss,) + output) if lm_loss is not None else output

#         return CausalLMOutputWithCrossAttentions(
#             loss=lm_loss,
#             logits=prediction_scores,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             cross_attentions=outputs.cross_attentions,
#         )

#     def prepare_inputs_for_generation(
#         self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **model_kwargs
#     ):
#         input_shape = input_ids.shape
#         # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
#         if attention_mask is None:
#             attention_mask = input_ids.new_ones(input_shape)

#         # cut decoder_input_ids if past_key_values is used
#         if past_key_values is not None:
#             past_length = past_key_values[0][0].shape[2]

#             # Some generation methods already pass only the last input ID
#             if input_ids.shape[1] > past_length:
#                 remove_prefix_length = past_length
#             else:
#                 # Default to old behavior: keep only final ID
#                 remove_prefix_length = input_ids.shape[1] - 1

#             input_ids = input_ids[:, remove_prefix_length:]

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "past_key_values": past_key_values,
#             "use_cache": use_cache,
#         }

#     def _reorder_cache(self, past_key_values, beam_idx):
#         reordered_past = ()
#         for layer_past in past_key_values:
#             reordered_past += (
#                 tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
#             )
#         return reordered_past


# @add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class BertLSHForMaskedLM(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertLSHModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


# @add_start_docstrings(
#     """Bert Model with a `next sentence prediction (classification)` head on top.""",
#     BERT_START_DOCSTRING,
# )
# class BertLSHForNextSentencePrediction(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)

#         self.bert = BertLSHModel(config)
#         self.cls = BertOnlyNSPHead(config)

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs,
#     ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
#             (see `input_ids` docstring). Indices should be in `[0, 1]`:

#             - 0 indicates sequence B is a continuation of sequence A,
#             - 1 indicates sequence B is a random sequence.

#         Returns:

#         Example:

#         ```python
#         >>> from transformers import AutoTokenizer, BertForNextSentencePrediction
#         >>> import torch

#         >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#         >>> model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

#         >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
#         >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
#         >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

#         >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
#         >>> logits = outputs.logits
#         >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
#         ```
#         """

#         if "next_sentence_label" in kwargs:
#             warnings.warn(
#                 "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
#                 " `labels` instead.",
#                 FutureWarning,
#             )
#             labels = kwargs.pop("next_sentence_label")

#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = outputs[1]

#         seq_relationship_scores = self.cls(pooled_output)

#         next_sentence_loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

#         if not return_dict:
#             output = (seq_relationship_scores,) + outputs[2:]
#             return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

#         return NextSentencePredictorOutput(
#             loss=next_sentence_loss,
#             logits=seq_relationship_scores,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# @add_start_docstrings(
#     """
#     Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
#     output) e.g. for GLUE tasks.
#     """,
#     BERT_START_DOCSTRING,
# )
class BertLSHForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertLSHModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# @add_start_docstrings(
#     """
#     Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
#     softmax) e.g. for RocStories/SWAG tasks.
#     """,
#     BERT_START_DOCSTRING,
# )
# class BertLSHForMultipleChoice(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)

#         self.bert = BertLSHModel(config)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = nn.Linear(config.hidden_size, 1)

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=MultipleChoiceModelOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
#             num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
#             `input_ids` above)
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

#         input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
#         attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
#         token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
#         position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
#         inputs_embeds = (
#             inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
#             if inputs_embeds is not None
#             else None
#         )

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         reshaped_logits = logits.view(-1, num_choices)

#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(reshaped_logits, labels)

#         if not return_dict:
#             output = (reshaped_logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return MultipleChoiceModelOutput(
#             loss=loss,
#             logits=reshaped_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# @add_start_docstrings(
#     """
#     Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
#     Named-Entity-Recognition (NER) tasks.
#     """,
#     BERT_START_DOCSTRING,
# )
# class BertLSHForTokenClassification(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.bert = BertLSHModel(config, add_pooling_layer=False)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
#         output_type=TokenClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
#         expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
#     )
#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]

#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)

#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return TokenClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# @add_start_docstrings(
#     """
#     Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
#     layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
#     """,
#     BERT_START_DOCSTRING,
# )
class BertLSHForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertLSHModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
