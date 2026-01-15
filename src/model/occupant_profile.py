from pydantic import BaseModel, Field
from pathlib import Path
import json
from model.occupancy import Occupancy
from model.lighting import Lighting
from model.equipment import Equipment
from model.hvac import HVAC

class Occupant(BaseModel):
    occupancy: Occupancy = Field(...)
    lighting: Lighting = Field(...)
    equipment: Equipment = Field(...)
    hvac: HVAC = Field(...)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "Occupant":
        """Load occupancy behavior from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(
            occupancy=Occupancy.model_validate(data["occupancy"]),
            lighting=Lighting.model_validate(data["lighting"]),
            equipment=Equipment.model_validate(data["equipment"]),
            hvac=HVAC.model_validate(data["hvac"])
        )
    
    def to_json_file(self, path: str | Path) -> None:
        """Save occupancy behavior to a JSON file."""
        data = {
            "occupancy": self.occupancy.model_dump(),
            "lighting": self.lighting.model_dump(),
            "equipment": self.equipment.model_dump(),
            "hvac": self.hvac.model_dump()
        }
        Path(path).write_text(json.dumps(data, indent=4))

