from functools import lru_cache
from typing import List

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration values for the Agentic Blog Writer.
    Fields WITHOUT a default are REQUIRED — app will refuse to start if missing.
    Fields WITH a default are optional and can be overridden in .env.
    """

    GROQ_API_KEY: str                            
    GROQ_MODEL_NAME: str = "llama3-70b-8192"     
    GROQ_TEMPERATURE: float = 0.3               
    GROQ_MAX_TOKENS: int = 4096                  

  
    TAVILY_API_KEY: str                          
    TAVILY_MAX_RESULTS: int = 5                  
    TAVILY_SEARCH_DEPTH: str = "advanced"        

    APP_ENV: str = "development"                 
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_TITLE: str = "Agentic Blog Writer"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "A multi-agent blog writing system powered by Groq."
    )
    LOG_LEVEL: str = "INFO"                      

    MAX_REVISION_CYCLES: int = 3               
    EVALUATION_THRESHOLD: float = 7.0            
    TOOL_TIMEOUT_SECONDS: int = 30              
    RESEARCHER_MAX_RESULTS: int = 5              
   
    RATE_LIMIT_GENERATE: str = "5/minute"        
    RATE_LIMIT_STATUS: str = "30/minute"         
    RATE_LIMIT_RESULT: str = "30/minute"         

   
    BLOCKED_KEYWORDS: List[str] = [
        "illegal",
        "harmful",
        "violence",
        "drug synthesis",
        "weapon",
        "exploit",
        "hack tutorial",
    ]
    TOPIC_MIN_LENGTH: int = 10                   
    TOPIC_MAX_LENGTH: int = 200                 

    
    API_SECRET_KEY: str = ""

   
    ALLOWED_ORIGINS: List[str] = ["*"]           
    
    JOB_TTL_HOURS: int = 24                      
    JOB_CLEANUP_INTERVAL_MINUTES: int = 60       

    
    BLOG_MIN_WORDS: int = 500                    
    BLOG_MAX_WORDS: int = 5000                   

    
    model_config = SettingsConfigDict(
        env_file=".env",                         
        env_file_encoding="utf-8",
        case_sensitive=True,                     
        extra="ignore",                          
    )

   

    @field_validator("APP_ENV")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        """Ensure APP_ENV is one of the allowed values."""
        allowed = {"development", "production", "testing"}
        if v not in allowed:
            raise ValueError(
                f"APP_ENV must be one of {allowed}. Got: '{v}'"
            )
        return v

    @field_validator("GROQ_TEMPERATURE")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """LLM temperature must be between 0.0 and 1.0."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"GROQ_TEMPERATURE must be between 0.0 and 1.0. Got: {v}"
            )
        return v

    @field_validator("EVALUATION_THRESHOLD")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Score threshold must be between 1.0 and 10.0."""
        if not (1.0 <= v <= 10.0):
            raise ValueError(
                f"EVALUATION_THRESHOLD must be between 1.0 and 10.0. Got: {v}"
            )
        return v

    @field_validator("MAX_REVISION_CYCLES")
    @classmethod
    def validate_max_revisions(cls, v: int) -> int:
        """Revision cycles must be between 1 and 10 to avoid infinite loops."""
        if not (1 <= v <= 10):
            raise ValueError(
                f"MAX_REVISION_CYCLES must be between 1 and 10. Got: {v}"
            )
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is a valid Python logging level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(
                f"LOG_LEVEL must be one of {allowed}. Got: '{v}'"
            )
        return v

    @field_validator("TAVILY_SEARCH_DEPTH")
    @classmethod
    def validate_search_depth(cls, v: str) -> str:
        """Tavily search depth must be 'basic' or 'advanced'."""
        allowed = {"basic", "advanced"}
        if v not in allowed:
            raise ValueError(
                f"TAVILY_SEARCH_DEPTH must be 'basic' or 'advanced'. Got: '{v}'"
            )
        return v

    

    @model_validator(mode="after")
    def validate_word_count_range(self) -> "Settings":
        """BLOG_MIN_WORDS must always be strictly less than BLOG_MAX_WORDS."""
        if self.BLOG_MIN_WORDS >= self.BLOG_MAX_WORDS:
            raise ValueError(
                f"BLOG_MIN_WORDS ({self.BLOG_MIN_WORDS}) must be less than "
                f"BLOG_MAX_WORDS ({self.BLOG_MAX_WORDS})."
            )
        return self


    @property
    def is_development(self) -> bool:
        """True when running in development mode."""
        return self.APP_ENV == "development"

    @property
    def is_production(self) -> bool:
        """True when running in production mode."""
        return self.APP_ENV == "production"

    @property
    def is_api_key_required(self) -> bool:
        """True if a static API secret key has been configured."""
        return bool(self.API_SECRET_KEY.strip())




@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the Settings singleton.

    @lru_cache ensures the .env file is read EXACTLY ONCE at startup,
    no matter how many modules call get_settings().

    In tests, call get_settings.cache_clear() to force a fresh reload:
        from app.config.settings import get_settings
        get_settings.cache_clear()
    """
    return Settings()