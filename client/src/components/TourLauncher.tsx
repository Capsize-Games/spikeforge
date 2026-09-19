import { GraduationCap } from "lucide-react";

import { HELP } from "../helpText";
import { useI18n } from "../i18n/I18nProvider";
import type { TourLesson } from "../tour/types";
import { HelpTip } from "./HelpTip";
import { IconButton } from "./IconButton";

interface Props {
  lessons: TourLesson[];
  open: boolean;
  onToggle: () => void;
  onOpen: (id: string) => void;
}

/** "Guided tours" header control and the lesson menu it reveals. */
export function TourLauncher({ lessons, open, onToggle, onOpen }: Props) {
  const { t } = useI18n();
  return (
    <div className="tour-launch-wrap">
      <IconButton
        icon={GraduationCap}
        label={t("tour.title")}
        onClick={onToggle}
        expanded={open}
        menu
      />
      <HelpTip text={HELP.tour} />

      {open && (
        <div className="tour-menu" role="menu">
          <div className="tour-menu-head">{t("tour.intro")}</div>
          {lessons.map((lesson) => (
            <button
              key={lesson.id}
              type="button"
              className="tour-menu-item"
              role="menuitem"
              onClick={() => onOpen(lesson.id)}
            >
              <span className="tour-menu-name">{lesson.name}</span>
              <span className="tour-menu-summary">{lesson.summary}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
