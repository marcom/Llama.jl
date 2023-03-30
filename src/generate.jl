# rough translation of examples/main/main.cpp
#
# TODO
# - interactive mode
# - instruct mode
# - input prefix
# - antiprompt
function generate(ctx::LlamaContext, prompt::AbstractString;
                  # these are already in ctx.params or ctx.model_path
                  #   seed::Int=-1,           # RNG seed
                  #   n_parts::Int=-1,        # amount of model parts, -1 = determine from model dims.
                  #   n_ctx::Int=512,         # context size
                  #   # memory_f16 == ctx.params.f16_kv
                  # model = "ggml.bin"
                  verbose::Bool=true,
                  n_threads::Int=1,       # number of threads to run on
                  n_predict::Int=128,     # new tokens to predict
                  repeat_last_n::Int=64,  # last n tokens to penalize
                  n_batch::Int=8,         # batch size for prompt processing
                  n_keep::Int=0,          # number of tokens to keep from initial prompt
                  # sampling parameters
                  top_k::Int=40,
                  top_p::Float64=0.95,
                  temp::Float64=0.80,
                  repeat_penalty::Float64=1.10,
                  # input_prefix = ""        # string to prefix user inputs with
                  # antiprompt = String[]    # string upon seeing which more user input is prompted
                  #interactive::Bool=false,   # interactive mode
                  #instruct::Bool=false,      # instruction mode (used for Alpaca models)
                  ignore_eos::Bool=false,    # do not stop generating after eos
                  )
    # names from main.cpp
    # - embd_inp => toks_inp
    # - embd     => toks

    #n_parts = ctx.params.n_parts
    n_ctx = ctx.params.n_ctx
    # memory_f16 = ctx.params.f16_kv

    @show toks_inp = tokenize(ctx, prompt)
    n_tok_inp = length(toks_inp)
    max_tok_inp = ctx.n_ctx - 4  # TODO: magic number 4
    if n_tok_inp > max_tok_inp
        error("input prompt is too long ($n_tok_inp tokens, max $max_tok_inp)")
    end

    if n_keep < 0 || n_keep > n_tok_inp
        n_keep = n_tok_inp
    end

    tok_newline = tokenize(ctx, "\n")

    if verbose
        println("prompt = $prompt")
        println("number of tokens in prompt = $n_tok_inp")
        for i = 1:n_tok_inp
            println("$(toks_inp[i]) -> $(token_to_str(ctx, toks_inp[i]))")
        end
        if n_keep > 0
            println("static prompt based on n_keep")
            for i = 1:n_keep
                print(token_to_str(ctx, toks_inp[i]))
            end
            println()
        end
        println()
    end
    println("sampling: temp = $temp, top_k = $top_k, top_p = $top_p," *
        " repeat_last_n = $repeat_last_n, repeat_penalty = $repeat_penalty")
    println("generate: n_ctx = $n_ctx, n_batch = $n_batch, n_predict = $n_predict, n_keep = $n_keep")
    println()

    last_n_tokens = zeros(LibLlama.llama_token, n_ctx)
    n_past = 0
    n_remain = n_predict
    n_consumed = 0

    toks = LibLlama.llama_token[]
    while n_remain > 0
        if length(toks) > 0
            # infinite text generation via context swapping
            # if we run out of context:
            # - take the n_keep first tokens from the original prompt (via n_past)
            # - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
            if n_past + length(toks) > n_ctx
                n_left = n_past - n_keep
                n_past = n_keep
                # insert n_left/2 tokens at the start of toks from last_n_tokens
                idx_start_off = n_ctx - n_left/2 - length(toks)
                idx_end_off = length(toks)
                prepend!(toks, @view last_n_tokens[begin+idx_start_off:end-idx_end_off])
                ret = llama_eval(ctx, toks; n_past, n_threads)
                if ret != 0
                    error("failed to run llama_eval")
                end
            end
        end

        n_past += length(toks)
        resize!(toks, 0)

        if n_tok_inp <= n_consumed
            # out of user input, sample next token
            id = zero(LibLlama.llama_token)
            lg = logits(ctx)
            if ignore_eos
                lg[ctx.token_eos] = 0
            end
            id = LibLlama.llama_sample_top_p_top_k(
                ctx.ptr,
                (@view last_n_tokens[n_ctx - repeat_last_n + 1:end]),
                repeat_last_n, top_k, top_p, temp, repeat_penalty
            )
            deleteat!(last_n_tokens, 1)
            push!(last_n_tokens, id)

            # replace end of text token with newline token when in
            # interactive mode
            #if id == tok_newline
            # TODO
            #end

            # add new token to the context
            push!(toks, id)
            n_remain -= 1
        else
            # some user input remains from prompt or interaction,
            # forward it to processing
            while n_tok_inp > n_consumed
                push!(toks, toks_inp[n_consumed + 1])  # TODO: +1 for 1-based indexing
                deleteat!(last_n_tokens, 1)
                push!(last_n_tokens, toks_inp[n_consumed + 1])
                n_consumed += 1
                if length(toks) >= n_batch
                    break
                end
            end
        end

        # display text
        for id in toks
            print(token_to_str(ctx, id))
        end

        if length(toks) > 0 && toks[end] == ctx.token_eos
            println("[end of text]")
        end
    end
end
