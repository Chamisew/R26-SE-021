/* Build-stack marks (inline SVG, no image files).
   stackOf() reads the stack from the service NAME/LINK only — never the
   deployed platform. The backend additionally live-detects the runtime from
   /metrics payloads (served as config.stacks); link keywords win when both
   are present. */

export const STACK_NAMES = {
  node: "node.js",
  python: "python",
  go: "go",
  java: "java",
  ruby: "ruby",
  php: "php",
  other: "stack unknown",
};

const LINK_KEYWORDS = [
  ["node", ["node", "nodejs", "npm", "express", "fastify", "nestjs", "nextjs", "nuxt", "deno", "bun"]],
  ["python", ["python", "django", "flask", "fastapi", "py-", "-py", "pytorch", "pandas"]],
  ["go", ["golang", "-go-", "/go", "gin", "fiber"]],
  ["java", ["java", "spring", "quarkus", "micronaut"]],
  ["ruby", ["ruby", "rails"]],
  ["php", ["php", "laravel", "symfony"]],
];

export function stackOf(svc) {
  const blob = `${svc?.name || ""} ${svc?.base_url || ""}`.toLowerCase();
  for (const [stack, tokens] of LINK_KEYWORDS) {
    if (tokens.some((t) => blob.includes(t))) return stack;
  }
  return "other";
}

const MARKS = {
  node: (
    <>
      <polygon points="12,1.5 21.5,7 21.5,17 12,22.5 2.5,17 2.5,7" fill="#339933" />
      <polygon points="12,5.5 18,9 18,15 12,18.5 6,15 6,9" fill="#2a7a2a" />
    </>
  ),
  python: (
    <>
      <circle cx="12" cy="12" r="10" fill="#3776ab" />
      <path d="M12 2 a10 10 0 0 1 0 20 Z" fill="#ffd43b" />
      <circle cx="9" cy="10" r="1.4" fill="#fff" />
      <circle cx="15" cy="14" r="1.4" fill="#fff" />
    </>
  ),
  go: (
    <>
      <rect x="2" y="2" width="20" height="20" rx="5" fill="#00add8" />
      <circle cx="12" cy="12" r="5.4" fill="none" stroke="#fff" strokeWidth="2.4" />
      <circle cx="12" cy="12" r="1.6" fill="#fff" />
    </>
  ),
  java: (
    <>
      <rect x="2" y="2" width="20" height="20" rx="5" fill="#f89820" />
      <path
        d="M9 7.5 h4.5 v6.5 a2.5 2.5 0 0 1 -5 0 Z M13.5 9.5 h1.6 a1.8 1.8 0 0 1 0 3.6 h-1.6"
        fill="none"
        stroke="#fff"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </>
  ),
  ruby: (
    <>
      <polygon points="12,2.5 20,9.5 12,21.5 4,9.5" fill="#cc342d" />
      <path d="M4 9.5 H20 M12 2.5 L8.5 9.5 L12 21.5 M12 2.5 L15.5 9.5 L12 21.5" stroke="#fff" strokeWidth="1.4" fill="none" />
    </>
  ),
  php: (
    <>
      <ellipse cx="12" cy="12" rx="10" ry="7.5" fill="#777bb4" />
      <ellipse cx="12" cy="12" rx="6" ry="4.2" fill="none" stroke="#fff" strokeWidth="1.8" />
    </>
  ),
  other: (
    <>
      <rect x="2" y="2" width="20" height="20" rx="5" fill="#64748b" />
      <circle cx="12" cy="12" r="3.4" fill="#fff" />
    </>
  ),
};

/* Deploy-platform fallback marks (used only when the build stack is unknown) */
const PLAT_MARKS = {
  azure: (
    <>
      <path d="M12 2.5 L21.5 20.5 L2.5 20.5 Z" fill="#0078d4" />
      <path d="M12 10 L16.8 18 L7.2 18 Z" fill="#fff" />
    </>
  ),
  render: (
    <>
      <rect x="2" y="2" width="20" height="20" rx="5" fill="#101828" />
      <path
        d="M9.5 6.5 v11 M9.5 6.5 H13 a3.2 3.2 0 0 1 0 6.4 H9.5 M12.6 12.9 L16 17.5"
        stroke="#0ea5e9"
        strokeWidth="2.2"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </>
  ),
  oracle: (
    <>
      <rect x="2" y="2" width="20" height="20" rx="5" fill="#c74634" />
      <ellipse cx="12" cy="12" rx="5.2" ry="6.8" fill="none" stroke="#fff" strokeWidth="2.4" />
    </>
  ),
  railway: (
    <>
      <rect x="2" y="2" width="20" height="20" rx="5" fill="#131318" />
      <path
        d="M7 13.5 L12 8.5 L17 13.5 M7 17.5 L12 12.5 L17 17.5"
        stroke="#fff"
        strokeWidth="2.2"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </>
  ),
};

export default function StackIcon({ stack, platform }) {
  const plat = String(platform || "other").toLowerCase();
  const mark =
    (stack && stack !== "other" && MARKS[stack]) || PLAT_MARKS[plat] || MARKS.other;
  return (
    <svg className="stack-ic" viewBox="0 0 24 24" aria-hidden="true">
      {mark}
    </svg>
  );
}
