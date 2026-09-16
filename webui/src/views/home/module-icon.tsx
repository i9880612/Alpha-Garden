import type { ReactNode } from "react";

const shapes = {
  run: (
    <>
      <circle cx="12" cy="12" r="8.5" fill="currentColor" fillOpacity=".08" strokeOpacity=".6" />
      <path d="m10 8.5 5 3.5-5 3.5z" fill="currentColor" strokeWidth="1.5" />
    </>
  ),
  library: (
    <>
      <path d="M16 7V5a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h1" strokeOpacity=".55" />
      <rect x="7" y="7" width="13" height="14" rx="3" fill="currentColor" fillOpacity=".08" />
      <path d="M11 12h5m-5 4h3" />
    </>
  ),
  submit: (
    <>
      <path
        d="m20.5 3.5-5.8 16.4a.8.8 0 0 1-1.5 0L10 14l-5.9-3.2a.8.8 0 0 1 0-1.5Z"
        fill="currentColor"
        fillOpacity=".1"
      />
      <path d="m10 14 6-6" />
    </>
  ),
  settings: (
    <>
      <path d="M5 3v5m0 4v9M12 3v10m0 4v4M19 3v3m0 4v11" strokeOpacity=".6" />
      <circle cx="5" cy="10" r="2" fill="currentColor" fillOpacity=".14" />
      <circle cx="12" cy="15" r="2" fill="currentColor" fillOpacity=".14" />
      <circle cx="19" cy="8" r="2" fill="currentColor" fillOpacity=".14" />
    </>
  ),
} satisfies Record<string, ReactNode>;

export function ModuleIcon({ kind }: { kind: keyof typeof shapes }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {shapes[kind]}
    </svg>
  );
}
