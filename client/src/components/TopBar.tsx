/** App header: a compact 24x24 logo mark, no title text. */
export function TopBar() {
  return (
    <header className="topbar">
      <span className="logo" role="img" aria-label="SNN Interpreter">
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinecap="round"
        >
          <circle cx="4" cy="6" r="1.7" fill="currentColor" stroke="none" />
          <circle cx="4" cy="12" r="1.7" fill="currentColor" stroke="none" />
          <circle cx="4" cy="18" r="1.7" fill="currentColor" stroke="none" />
          <circle cx="12" cy="9" r="1.7" fill="currentColor" stroke="none" />
          <circle cx="12" cy="15" r="1.7" fill="currentColor" stroke="none" />
          <circle cx="20" cy="12" r="1.7" fill="currentColor" stroke="none" />
          <path d="M5.7 6.5 10.3 8.6M5.7 11.4 10.3 9.4M5.7 12.6 10.3 14.6M5.7 17.6 10.3 15.4M13.7 9.4 18.3 11.4M13.7 14.6 18.3 12.6" />
        </svg>
      </span>
    </header>
  );
}
