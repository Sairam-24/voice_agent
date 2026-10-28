// Display helpers. Nothing here invents a value: a missing number is shown as a dash.

/** 12.3 -> "12 ms", 1234 -> "1.23 s", null -> "—". The space is non-breaking so a number never wraps from its unit. */
export function formatMs(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${Math.round(value)} ms`;
}

/** 75 -> "1:15" */
export function formatClock(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/** Each language in its own script, as a person looks for it in a menu. */
const LANGUAGE_LABELS: Record<string, string> = {
  auto: "Auto",
  en: "English",
  te: "తెలుగు",
  hi: "हिन्दी",
  ta: "தமிழ்",
  kn: "ಕನ್ನಡ",
  ml: "മലയാളം",
  mr: "मराठी",
  bn: "বাংলা",
  gu: "ગુજરાતી",
  pa: "ਪੰਜਾਬੀ",
  ur: "اردو",
  es: "Español",
  fr: "Français",
  de: "Deutsch",
};

export function languageLabel(code: string): string {
  return LANGUAGE_LABELS[code] ?? code.toUpperCase();
}

/** "12 pages", "4k characters"; "(only the start)" when the server kept only part of a long file. */
export function describeFile(file: { chars?: number; pages?: number | null; truncated?: boolean }): string {
  const size =
    file.pages != null
      ? `${file.pages} page${file.pages === 1 ? "" : "s"}`
      : file.chars !== undefined
        ? `${file.chars < 1000 ? file.chars : `${Math.round(file.chars / 1000)}k`} characters`
        : "";
  return file.truncated ? `${size} (only the start)` : size;
}
