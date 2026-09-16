import alphaGardenLogoUrl from "@/assets/logo/logo-full.svg";

export default function AlphaLoading({ fullscreen = false, className = "" }: { fullscreen?: boolean; className?: string }) {
  return (
    <div className={`ag-loading ${fullscreen ? "ag-loading--fullscreen" : "ag-loading--compact"} ${className}`} role="status" aria-label="Loading">
      <img className="ag-loading-logo" src={alphaGardenLogoUrl} alt="Alpha Garden" width="220" height="64" />
      <span className="ag-loading-track" aria-hidden="true" />
      <span className="ag-loading-label">Loading</span>
    </div>
  );
}
