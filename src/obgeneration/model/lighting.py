from pydantic import BaseModel, Field


class Lighting(BaseModel):
    """
    Represents the lighting profile of a household.
    """
    if_led: bool = Field(..., description="Indicates if the household uses LED lighting.")
    #when_away: bool = Field(False, description="Indicates if lighting is on when people are away.")
    when_daylight_bright: bool = Field(True, description="Indicates if lighting is adjusted based on daylight brightness.")
