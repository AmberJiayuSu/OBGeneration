from obgeneration.generator.ob_generator import OccupantBehavior
from obgeneration.model.occupant_profile import Occupant
from obgeneration.generator.ob_utils import ScheduleUtils
from pathlib import Path
import json
import pytest
import time

class TestOverall:

    @pytest.mark.parametrize("input_file", [
        "test/unit/input/OB_1.json",
        "test/unit/input/OB_2.json",
        "test/unit/input/OB_3.json"
    ])
    def test_OB(self, input_file):
        start_time = time.time()
        print(f"\n=== Starting test for {input_file} ===")

        occupant = Occupant.from_json_file(Path(input_file))
        ob_annual = OccupantBehavior.to_OB_annual(15, occupant)

        elapsed_time = time.time() - start_time
        print(f"=== OB generation completed in {elapsed_time:.2f}s ===")

        if elapsed_time > 10.0:
            raise RuntimeError(f"Test exceeded 10 second timeout: took {elapsed_time:.2f}s")

        occupancy_schedule = ob_annual.occupancy_schedule
        ScheduleUtils.plot_typical_week_schedule(occupancy_schedule, "Occupancy Schedule", f"test/unit/output/overall/{Path(input_file).stem}_occupancy_schedule.png", 15)
        ScheduleUtils.plot_annual_schedule_heatmap(occupancy_schedule, "Occupancy Schedule", f"test/unit/output/overall/{Path(input_file).stem}_occupancy_schedule_heatmap.png", 15)
        lighting_schedule = ob_annual.lighting_schedule
        ScheduleUtils.plot_typical_week_schedule(lighting_schedule, "Lighting Schedule", f"test/unit/output/overall/{Path(input_file).stem}_lighting_schedule.png", 15)
        ScheduleUtils.plot_annual_schedule_heatmap(lighting_schedule, "Lighting Schedule", f"test/unit/output/overall/{Path(input_file).stem}_lighting_schedule_heatmap.png",  15, 'binary')
        equipment_schedule = ob_annual.equipment_schedule
        ScheduleUtils.plot_typical_week_schedule(equipment_schedule, "Equipment Schedule", f"test/unit/output/overall/{Path(input_file).stem}_equipment_schedule.png", 15,vmin=300,vmax=1800)
        ScheduleUtils.plot_annual_schedule_heatmap(equipment_schedule, "Equipment Schedule", f"test/unit/output/overall/{Path(input_file).stem}_equipment_schedule_heatmap.png", 15, 'Greys', vmin=300,vmax=1800)
        dhw_schedule = ob_annual.dhw_schedule
        ScheduleUtils.plot_annual_schedule_heatmap(dhw_schedule, "DHW Schedule", f"test/unit/output/overall/{Path(input_file).stem}_dhw_schedule_heatmap.png", 15)
        heating_setpoint = ob_annual.heating_setpoint
        if heating_setpoint is not None:
            ScheduleUtils.plot_typical_week_schedule(heating_setpoint, "Heating Setpoint", f"test/unit/output/overall/{Path(input_file).stem}_heating_setpoint.png", 15,vmin=5,vmax=25)
            ScheduleUtils.plot_annual_schedule_heatmap(heating_setpoint, "Heating Setpoint", f"test/unit/output/overall/{Path(input_file).stem}_heating_setpoint_heatmap.png", 15,'YlOrRd',vmin=5,vmax=25)
        cooling_setpoint = ob_annual.cooling_setpoint
        if cooling_setpoint is not None:
            ScheduleUtils.plot_typical_week_schedule(cooling_setpoint, "Cooling Setpoint", f"test/unit/output/overall/{Path(input_file).stem}_cooling_setpoint.png", 15,vmin=20,vmax=35)
            ScheduleUtils.plot_annual_schedule_heatmap(cooling_setpoint, "Cooling Setpoint", f"test/unit/output/overall/{Path(input_file).stem}_cooling_setpoint_heatmap.png", 15,'winter',vmin=20,vmax=35)
        
  