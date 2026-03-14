import sys

from loguru import logger

logger.remove()  # Supprime le handler par défaut
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="DEBUG"
)
logger.add(
    "logs/research_tool.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)
