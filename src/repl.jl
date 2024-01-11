global REPL_HAS_CTX::Bool = false
global REPL_CTX::Ref{LlamaContext}

function get_repl_llama()
    if REPL_HAS_CTX
        return REPL_CTX[]
    else
        return missing
    end
end

function set_repl_llama(ctx::LlamaContext)
    REPL_HAS_CTX = true
    REPL_CTX = ctx
    return nothing
end

function repl_llama(s)
    @warn "REPL Llama is not yet implemented. Please use `run_*` functions instead. See `?Llama.run_server` for more information."
    # TODO
    return s
end

function init_repl()
    if ismissing(get_repl_llama())
        @warn "REPL LLaMA model not set, please set with `Llama.set_repl_llama(ctx)`"
    end
    ReplMaker.initrepl(repl_llama,
        prompt_text = "LLaMA> ",
        prompt_color = :blue,
        start_key = '}',
        mode_name = "LLaMA_mode")
end
