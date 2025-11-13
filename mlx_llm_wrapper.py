"""
MLX LM wrapper for LangChain compatibility
"""

from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors


class MLXLLM(LLM):
    """MLX LM wrapper for LangChain"""
    
    model_path: str = Field(default="mlx-community/Qwen2-7B-Instruct-4bit")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2048)
    top_p: float = Field(default=0.8)
    repetition_penalty: float = Field(default=1.05)
    
    _model: Any = None
    _tokenizer: Any = None
    
    @property
    def _llm_type(self) -> str:
        return "mlx_lm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the MLX model"""
        if self._model is None or self._tokenizer is None:
            self._model, self._tokenizer = load(self.model_path)
        
        # Check if tokenizer has chat template (for Qwen and other chat models)
        if hasattr(self._tokenizer, 'apply_chat_template') and self._tokenizer.chat_template:
            # Format as chat message for better structured output
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        # Generate response using sampler for temperature and other parameters
        sampler = make_sampler(
            temp=self.temperature,
            top_p=self.top_p,
        )
        
        # Use logits processors for repetition penalty
        logits_processors = make_logits_processors(
            repetition_penalty=self.repetition_penalty,
        )
        
        response = generate(
            self._model,
            self._tokenizer,
            prompt=formatted_prompt,
            verbose=False,
            max_tokens=self.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        )
        
        return response
    
    def invoke(self, input: str, config: Optional[Any] = None, **kwargs: Any) -> Any:
        """Invoke the model (LangChain interface)"""
        from langchain_core.messages import AIMessage
        
        response_text = self._call(input, **kwargs)
        return AIMessage(content=response_text)

