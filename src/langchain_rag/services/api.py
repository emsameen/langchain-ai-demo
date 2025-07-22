"""API service connectors for external services."""

import os
from typing import Dict, List, Optional, Union

import google.generativeai as genai
from langchain_openai import ChatOpenAI, OpenAI
from langchain_huggingface import HuggingFaceEndpoint

from langchain_rag.utils.helpers import load_env_variables


class LLMService:
    """Service class for managing LLM providers."""
    
    def __init__(self, env_file: str = ".env"):
        """Initialize the LLM service.
        
        Args:
            env_file: Path to environment variables file
        """
        self.env_vars = load_env_variables(env_file)
        
        # Set API keys from environment variables
        if self.env_vars.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.env_vars["OPENAI_API_KEY"]
        
        if self.env_vars.get("HUGGINGFACE_API_KEY"):
            os.environ["HUGGINGFACE_API_KEY"] = self.env_vars["HUGGINGFACE_API_KEY"]
        
        if self.env_vars.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = self.env_vars["GOOGLE_API_KEY"]
            genai.configure(api_key=self.env_vars["GOOGLE_API_KEY"])
    
    def get_openai_llm(
        self, 
        model_name: str = "gpt-3.5-turbo", 
        temperature: float = 0.7,
        **kwargs
    ) -> Union[OpenAI, ChatOpenAI]:
        """Get an OpenAI LLM instance.
        
        Args:
            model_name: Name of the OpenAI model
            temperature: Temperature parameter for generation
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            OpenAI or ChatOpenAI instance
        """
        if not self.env_vars.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found in environment variables")
            
        if model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4"):
            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                **kwargs
            )
        else:
            return OpenAI(
                model_name=model_name,
                temperature=temperature,
                **kwargs
            )
    
    def get_huggingface_llm(
        self,
        model_name: str = "google/flan-t5-xxl",
        **kwargs
    ) -> HuggingFaceEndpoint:
        """Get a HuggingFace LLM instance.
        
        Args:
            model_name: Name of the HuggingFace model
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            HuggingFaceEndpoint instance
        """
        if not self.env_vars.get("HUGGINGFACE_API_KEY"):
            raise ValueError("HuggingFace API key not found in environment variables")
            
        return HuggingFaceEndpoint(
            repo_id=model_name,
            **kwargs
        )
    
    def get_gemini_llm(
        self,
        model_name: str = "gemini-pro",
        **kwargs
    ) -> ChatOpenAI:  # Actually returns a wrapper around Gemini
        """Get a Google Gemini LLM instance through LangChain integration.
        
        Args:
            model_name: Name of the Gemini model
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            ChatOpenAI instance configured for Gemini
        """
        if not self.env_vars.get("GOOGLE_API_KEY"):
            raise ValueError("Google API key not found in environment variables")
            
        # LangChain provides Google models through ChatOpenAI with appropriate model name
        return ChatOpenAI(
            model=model_name,
            **kwargs
        )
