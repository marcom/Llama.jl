# adapted from llama.cpp: examples/simple/simple.cpp

# using .LibLlama: llama_get_kv_cache_token_count, llama_get_logits,
#     llama_n_ctx, llama_n_vocab, llama_token, llama_tokenize,
#     llama_token_to_str

# TODO: needed API functions
# token2string(tok::llama_token) -> String
# token2string(tok::llama_token) = 

function generate(ctx::LlamaContext, prompt::AbstractString; nthreads::Int=1)
    token2str(tok::LibLlama.llama_token) = unsafe_string(LibLlama.llama_token_to_str(ctx.ptr, tok))
    max_context_size = LibLlama.llama_n_ctx(ctx.ptr)
    max_tokens_list_size = max_context_size - 4

    # convert prompt to tokens
    # TODO: how to check if prompt produced more than n_max_tokens?
    n_max_tokens = 4096
    tokens_list = zeros(LibLlama.llama_token, n_max_tokens)
    add_bos = true
    ntokens = LibLlama.llama_tokenize(ctx.ptr, prompt, tokens_list, n_max_tokens, add_bos)
    if ntokens < 0
        error("Failed to tokenize prompt: $prompt")
    end
    resize!(tokens_list, ntokens)

    # print prompt tokens
    for i = 1:ntokens
        println(token2str(tokens_list[i]))
    end
    flush(stdout)

    # main prediction loop
    # TODO: implement infinite text generation via context swapping, see main.cpp
    while LibLlama.llama_get_kv_cache_token_count(ctx.ptr) < max_context_size
        # evaluate tokens
        npast = LibLlama.llama_get_kv_cache_token_count(ctx.ptr)
        if LibLlama.llama_eval(ctx.ptr, tokens_list, length(tokens_list), npast, nthreads) != 0
            error("Failed to run llama_eval")
        end
        empty!(tokens_list)

        # select the best prediction
        new_token_id = LibLlama.llama_token(0)
        logits = LibLlama.llama_get_logits(ctx.ptr)
        n_vocab = LibLlama.llama_n_vocab(ctx.ptr)
        candidates = LibLlama.llama_token_data[]
        sizehint!(candidates, n_vocab)
        for token_id = 0:n_vocab-1
            push!(candidates, LibLlama.llama_token_data(
                LibLlama.llama_token(token_id),
                unsafe_load(logits, token_id + 1),
                0.0f0
            ))
        end
        sorted = false
        candidates_p = Ref(LibLlama.llama_token_data_array(pointer(candidates), length(candidates), sorted))
        new_token_id = LibLlama.llama_sample_token_greedy(ctx.ptr, candidates_p)

        # check for end of stream (eos)
        if new_token_id == LibLlama.llama_token_eos()
            println(stderr, "[end of text]")
            break
        end

        # print new token
        print(token2str(new_token_id))
        flush(stdout)

        push!(tokens_list, new_token_id)
    end

    #llama_free(ctx.ptr)
    #llama_free_model(model)
    #llama_backend_free()
end
