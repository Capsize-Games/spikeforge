import type { RasterPayload } from "../types";
import { HeatmapCanvas } from "./HeatmapCanvas";
import { RasterCanvas } from "./RasterCanvas";

interface Props {
  sample: number[][] | null;
  spikeFrame: number[][] | null;
  reconGain1: number[][] | null;
  reconLow: number[][] | null;
  raster: RasterPayload | null;
}

export function ViewerPanels({
  sample,
  spikeFrame,
  reconGain1,
  reconLow,
  raster,
}: Props) {
  return (
    <div className="col-viz">
      <div className="row">
        <HeatmapCanvas
          data={sample}
          palette="binary"
          label="Input sample"
          width={224}
          height={224}
        />
        <HeatmapCanvas
          data={spikeFrame}
          palette="plasma"
          label="Spike frame"
          width={224}
          height={224}
        />
        {(reconGain1 || reconLow) && (
          <div className="panel">
            <div className="panel-title">Reconstruction</div>
            <div className="pair">
              <HeatmapCanvas
                data={reconGain1}
                palette="binary"
                label="Gain=1"
                width={120}
                height={120}
              />
              <HeatmapCanvas
                data={reconLow}
                palette="binary"
                label="Low gain"
                width={120}
                height={120}
              />
            </div>
          </div>
        )}
      </div>

      <RasterCanvas raster={raster} label="Spike raster" />
    </div>
  );
}
