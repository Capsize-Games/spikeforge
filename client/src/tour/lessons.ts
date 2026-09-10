import { INPUT_LESSONS } from "./lessonsInput";
import { NETWORK_LESSONS } from "./lessonsNetwork";
import type { TourLesson } from "./types";

/** Every tutorial lesson, in launcher order. */
export const LESSONS: TourLesson[] = [
  ...INPUT_LESSONS,
  ...NETWORK_LESSONS,
];
