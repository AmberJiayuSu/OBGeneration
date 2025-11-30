from typing import Optional
from pydantic import BaseModel, Field, computed_field
from enum import Enum

class NumberRange(BaseModel):
    min: int = Field(default=0)
    max: int = Field(default=0)

class LaundryEquipment(BaseModel):
    has_washer: bool = Field(default=True)
    washer_efficient: bool = Field(default=True)
    has_dryer: bool = Field(default=False)
    dryer_efficient: bool = Field(default=True)
    usage_frequency_per_week: Optional[NumberRange] = Field(default=None)

class RefrigerationSize(Enum):
    MINI = "mini"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class RefrigerationEquipment(BaseModel):
    has_refrigerator: bool = Field(default=True)
    efficient_refrigerator: bool = Field(default=True)
    size: RefrigerationSize = Field(default=RefrigerationSize.MEDIUM)

class DishwasherEquipment(BaseModel):
    has_dishwasher: bool = Field(default=False)
    dishwasher_efficient: bool = Field(default=True)
    usage_frequency_per_week: Optional[NumberRange] = Field(default=None)


class Equipment(BaseModel):
    """
    Represents the equipment profile of a household.
    Attributes:
        has_washing_machine: Indicates if the household has a washing machine.
        has_dishwasher: Indicates if the household has a dishwasher.
    """
    laundry: LaundryEquipment = Field(default_factory=LaundryEquipment)
    refrigeration: RefrigerationEquipment = Field(default_factory=RefrigerationEquipment)
    dishwasher: DishwasherEquipment = Field(default_factory=DishwasherEquipment)
    