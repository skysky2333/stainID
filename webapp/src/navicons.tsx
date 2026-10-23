/* The folded menu's icons (App.tsx): one small line drawing per page, 24×24, drawn in the text colour. */

const PATHS: Record<string, string[]> = {
  '/': ['M3 11.5 12 4l9 7.5', 'M5.5 10v10h13V10', 'M10 20v-5.5h4V20'],
  '/tutorials': ['M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18z', 'M10 8.5 16 12l-6 3.5z'],
  '/setup': ['M4 6h9', 'M17 6h3', 'M4 12h3', 'M11 12h9', 'M4 18h11', 'M19 18h1', 'M15 4v4', 'M9 10v4', 'M17 16v4'],
  '/workflow': ['M7 4.5 19 12 7 19.5z'],
  '/cohort': ['M7 4a3 3 0 1 0 0 6a3 3 0 1 0 0-6z', 'M17 4a3 3 0 1 0 0 6a3 3 0 1 0 0-6z', 'M7 14a3 3 0 1 0 0 6a3 3 0 1 0 0-6z',
    'M17 14a3 3 0 1 0 0 6a3 3 0 1 0 0-6z'],
  '/calibration': ['M12 3c3.2 4.4 6 7.9 6 11a6 6 0 0 1-12 0c0-3.1 2.8-6.6 6-11z', 'M9 15h6'],
  '/results': ['M3 20h18', 'M6 20v-8', 'M12 20V5', 'M18 20v-5'],
  '/labels': ['M3 4h8l10 10-7 7L4 11z', 'M7.5 6.5a1.5 1.5 0 1 0 0 3a1.5 1.5 0 1 0 0-3z'],
  '/models': ['M12 3 21 8l-9 5-9-5z', 'M3 12.5l9 5 9-5', 'M3 16.5l9 5 9-5'],
  '/users': ['M9 5a3 3 0 1 0 0 6a3 3 0 1 0 0-6z', 'M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6', 'M16 5.3a3 3 0 0 1 0 5.4', 'M17.5 14.4c2 .7 3.5 2.9 3.5 5.6'],
  '/settings': ['M12 8.5a3.5 3.5 0 1 0 0 7a3.5 3.5 0 1 0 0-7z', 'M12 2.5v3', 'M12 18.5v3', 'M2.5 12h3', 'M18.5 12h3', 'M5.3 5.3l2.1 2.1',
    'M16.6 16.6l2.1 2.1', 'M5.3 18.7l2.1-2.1', 'M16.6 7.4l2.1-2.1'],
  '/help': ['M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18z', 'M9.5 9.5a2.5 2.5 0 1 1 3.6 2.2c-.7.4-1.1 1-1.1 1.8v.5', 'M12 17v.5'],
  '/account': ['M12 4a4 4 0 1 0 0 8a4 4 0 1 0 0-8z', 'M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8'],
  project: ['M3 6.5h6l2 2h10V19H3z'],
}
const OTHER = ['M12 4a8 8 0 1 0 0 16a8 8 0 1 0 0-16z']

/** The icon of a page of the menu (its address) or of the project switcher ("project"). */
export function NavIcon({ of }: { of: string }) {
  return (
    <svg className="nav-icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      {(PATHS[of] ?? OTHER).map((d) => <path key={d} d={d} />)}
    </svg>
  )
}
