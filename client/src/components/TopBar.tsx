import { Activity } from "lucide-react";

import { useTheme } from "../theme";
import type { TourLesson } from "../tour/types";
import type { ExecutionMode } from "../types";
import { useI18n } from "../i18n/I18nProvider";
import { LanguageSelect } from "./LanguageSelect";
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
  const { t } = useI18n();
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

      <LanguageSelect />

      <button
        type="button"
        className="icon-btn theme-toggle"
        onClick={toggle}
        title={theme === "dark" ? t("theme.light") : t("theme.dark")}
        aria-label={t("theme.toggle")}
      >
        {theme === "dark" ? "☀" : "☾"}
      </button>
    </header>
  );
}
