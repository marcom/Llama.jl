global REPL_HAS_CTX::Bool = true
global REPL_CTX::LlamaContext = LlamaContext()

function get_repl_llama()
    if REPL_HAS_CTX
        return REPL_CTX
    else
        return missing
    end
end

function set_repl_llama(ctx::LlamaContext)
    REPL_HAS_CTX = true
    REPL_CTX = ctx
end

function repl_llama(s)
    # TODO
    return s
end

function init_repl()
    if ismissing(get_repl_llama())
        @warn "REPL LLaMA model not set, please set with `Llama.set_repl_llama(ctx)`"
    end
    ReplMaker.initrepl(
        repl_llama,
        prompt_text = "LLaMA> ",
        prompt_color = :blue,
        start_key = '}',
        mode_name = "LLaMA_mode",
    )
end
