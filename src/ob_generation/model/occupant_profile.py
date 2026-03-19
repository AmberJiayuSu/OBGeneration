from typing import Optional
from pydantic import BaseModel, Field
from pathlib import Path
import json
from ob_generation.model.occupancy_old import Occupancy
from ob_generation.model.lighting import Lighting
from ob_generation.model.equipment import Equipment
from ob_generation.model.hvac import HVAC
from ob_generation.model.window import Window

class Occupant(BaseModel):
    occupancy: Occupancy = Field(...)
    lighting: Lighting = Field(...)
    equipment: Equipment = Field(...)
    hvac: HVAC = Field(...)
    window: Window = Field(...)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "Occupant":
        """Load occupancy behavior from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(
            occupancy=Occupancy.model_validate(data["occupancy"])if "occupancy" in data else Occupancy(),
            lighting=Lighting.model_validate(data["lighting"]) if "lighting" in data else Lighting(),
            equipment=Equipment.model_validate(data["equipment"]) if "equipment" in data else Equipment(),
            hvac=HVAC.model_validate(data["hvac"]) if "hvac" in data else HVAC(),
            window=Window.model_validate(data["window"]) if "window" in data else Window()
        )

    def to_json_file(self, path: str | Path) -> None:
        """Save occupancy behavior to a JSON file."""
        data = {
            "occupancy": self.occupancy.model_dump(),
            "lighting": self.lighting.model_dump(),
            "equipment": self.equipment.model_dump(),
            "hvac": self.hvac.model_dump(),
            "window": self.window.model_dump()
        }
        Path(path).write_text(json.dumps(data, indent=4))

