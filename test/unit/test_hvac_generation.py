import pytest
import numpy as np

from generator.occupancy_generator import OccupancyGenerator, OccupancyAssumptions
from model.occupancy import Occupancy
from model.hvac import HVAC
from generator.hvac_generator import HVACAssumptions, HVACGenerator


class TestHVACGeneration:

    @pytest.mark.parametrize("occ", ["occ_1"], indirect=True)
    def test_no_hvac(self, occ):
        """Test case where there is no HVAC system."""
        hvac_json = """
        {
            "heating": null,
            "cooling": null
        }"""

        hvac = HVAC.model_validate_json(hvac_json)

        hvac_generator = HVACGenerator(hvac, HVACAssumptions.default())

        occ1 = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy,sleep=occ1.household_annual_schedule()
        occupancy_mask,sleep_mask,_ = occ1.get_annual_mask(occupancy,sleep)

        heating = hvac_generator.heating_setpoint_annual_schedule(occupancy_mask, sleep_mask)
        cooling = hvac_generator.cooling_setpoint_annual_schedule(occupancy_mask, sleep_mask)

        assert heating is None
        assert cooling is None

    @pytest.mark.parametrize("occ", ["occ_1"], indirect=True)
    def test_heating_no_control(self, occ):
        """Test case where there is heating but no control schedule."""
        hvac_json = """
        {
            "heating": {
                "type": "no_control"
            },
            "cooling": null
        }"""

        hvac = HVAC.model_validate_json(hvac_json)

        hvac_generator = HVACGenerator(hvac, HVACAssumptions.default())

        occ1 = OccupancyGenerator(occ, OccupancyAssumptions.default())
        occupancy,sleep=occ1.household_annual_schedule()
        occupancy_mask,sleep_mask,_ = occ1.get_annual_mask(occupancy,sleep)

        heating = hvac_generator.heating_setpoint_annual_schedule(occupancy_mask, sleep_mask)
        cooling = hvac_generator.cooling_setpoint_annual_schedule(occupancy_mask, sleep_mask)

        assert heating is not None
        assert len(heating) == len(occupancy_mask)  
        assert cooling is None