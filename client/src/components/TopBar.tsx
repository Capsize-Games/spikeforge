import { Activity } from "lucide-react";

import { useTheme } from "../theme";
import type { TourLesson } from "../tour/types";
import type { ExecutionMode } from "../types";
import { ModeToggle } from "./ModeToggle";
import { TourLauncher } from "./TourLauncher";

interface Props {
  mode: ExecutionMode;
  onModeChange: (mode: ExecutionMode) => void;
  lessons: TourLesson[];
  tourOpen: boolean;
  onToggleTours: () => void;
  onOpenTour: (id: string) => void;
}

/** App header: logo, guided tours, execution-mode toggle, theme toggle. */
export function TopBar({
  mode,
  onModeChange,
  lessons,
  tourOpen,
  onToggleTours,
  onOpenTour,
}: Props) {
  const { theme, toggle } = useTheme();
  return (
    <header className="topbar">
      <span className="logo" role="img" aria-label="Spikeforge">
        <Activity size={22} strokeWidth={2} />
      </span>

      <ModeToggle mode={mode} onChange={onModeChange} />

      <TourLauncher
        lessons={lessons}
        open={tourOpen}
        onToggle={onToggleTours}
        onOpen={onOpenTour}
      />

      <button
        type="button"
        className="icon-btn theme-toggle"
        onClick={toggle}
        title={
          theme === "dark" ? "Switch to light theme" : "Switch to dark theme"
        }
        aria-label="Toggle color theme"
      >
        {theme === "dark" ? "☀" : "☾"}
      </button>
    </header>
  );
}
