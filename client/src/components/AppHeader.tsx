import type { ExecutionMode, ModelLoadedPayload } from "../types";
import type { TabId } from "../tabs";
import { TAB_LABEL_KEYS } from "../tabLabels";
import { useI18n } from "../i18n/I18nProvider";
import { TopBar } from "./TopBar";
import type { SessionFacts } from "./SessionContext";

interface Props {
  mode: ExecutionMode;
  onModeChange: (mode: ExecutionMode) => void;
  /** The workspace (section) currently shown, so the toolbar can name it. */
  active: TabId;
  /** Session facts read from App state; an unknown value renders nothing. */
  loaded: ModelLoadedPayload | null;
  dataset: string;
  tourOpen: boolean;
  onToggleTours: () => void;
  onOpenTour: (id: string) => void;
}

/**
 * The fixed toolbar. It states where the session is (the active workspace) and
 * what it is working on (the loaded checkpoint and the dataset).
 *
 * The checkpoint's own numbers — accuracy, input mode, hidden width, device —
 * are summarised in the network inspector instead, which is where they are
 * read and where the device is set; a toolbar is not the place for them.
 */
export function AppHeader({
  mode,
  onModeChange,
  active,
  loaded,
  dataset,
  tourOpen,
  onToggleTours,
  onOpenTour,
}: Props) {
  const { t } = useI18n();
  const session: SessionFacts = {
    model: loaded?.name ?? null,
    dataset,
  };

  return (
    <header className="app-header">
      <TopBar
        mode={mode}
        onModeChange={onModeChange}
        workspace={t(TAB_LABEL_KEYS[active])}
        session={session}
        tourOpen={tourOpen}
        onToggleTours={onToggleTours}
        onOpenTour={onOpenTour}
      />
    </header>
  );
}
