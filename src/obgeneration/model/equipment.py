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



class FuelType(Enum):
    ELECTRIC = "electric"
    GAS = "gas"
    OTHER = "other"

class DishwashingPattern(Enum):
    AFTER_EACH_COOKED_MEAL = "after_each_cooked_meal"
    DAILY_BATCH_IF_COOKED = "daily_batch_if_cooked"
    WHENEVER_FULL = "whenever_full"
    INDEPENDENT_FREQUENCY = "independent_frequency"

class DishwashingLogic(BaseModel):
    pattern_type: DishwashingPattern = Field(default=DishwashingPattern.DAILY_BATCH_IF_COOKED)
    usage_frequency_per_week: Optional[NumberRange] = Field(default=None)


class RefrigerationEquipment(BaseModel):
    has_refrigerator: bool = Field(default=True)
    efficient_refrigerator: bool = Field(default=True)
    number_of_refrigerators: int = Field(default=1, ge=0)

class DishwasherEquipment(BaseModel):
    has_dishwasher: bool = Field(default=False)
    dishwasher_efficient: bool = Field(default=True)
    dishwashing_operational_logic: DishwashingLogic = Field(default_factory=DishwashingLogic)

class CookingEquipment(BaseModel):
    has_cooking_products: bool = Field(default=True)
    cooking_products_fuel: FuelType = Field(default=FuelType.ELECTRIC)
    usage_frequency_per_week: Optional[NumberRange] = Field(default=None)


class Equipment(BaseModel):
    """
    Represents the equipment profile of a household.
    Attributes:
        laundry (LaundryEquipment): Laundry equipment details.
        refrigeration (RefrigerationEquipment): Refrigeration equipment details.
        dishwasher (DishwasherEquipment): Dishwasher equipment details.

    """
    laundry: LaundryEquipment = Field(default_factory=LaundryEquipment)
    refrigerator: RefrigerationEquipment = Field(default_factory=RefrigerationEquipment)
    dishwasher: DishwasherEquipment = Field(default_factory=DishwasherEquipment)
    cooking_products: CookingEquipment = Field(default_factory=CookingEquipment)
    