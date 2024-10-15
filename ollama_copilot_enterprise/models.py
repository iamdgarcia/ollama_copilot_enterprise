from pydantic import BaseModel, Field


# Data model
class ManimCode(BaseModel):
    """Schema for code solutions to questions about LCEL."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    scene_name: str = Field(description="Name of the scene to being rendered")
