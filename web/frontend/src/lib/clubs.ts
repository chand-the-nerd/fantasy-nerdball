/**
 * Club identity strip for the shirt cards.
 *
 * `band` is the primary shirt colour, `trim` the secondary. Keys match the
 * names the FPL API returns, with a few aliases for how the optimiser
 * abbreviates them.
 */
interface ClubColours {
  band: string;
  trim: string;
  abbr: string;
}

const CLUBS: Record<string, ClubColours> = {
  Arsenal: { band: "#EF0107", trim: "#FFFFFF", abbr: "ARS" },
  "Aston Villa": { band: "#95BFE5", trim: "#670E36", abbr: "AVL" },
  Bournemouth: { band: "#DA291C", trim: "#000000", abbr: "BOU" },
  Brentford: { band: "#E30613", trim: "#FFFFFF", abbr: "BRE" },
  Brighton: { band: "#0057B8", trim: "#FFCD00", abbr: "BHA" },
  Burnley: { band: "#6C1D45", trim: "#99D6EA", abbr: "BUR" },
  Chelsea: { band: "#034694", trim: "#FFFFFF", abbr: "CHE" },
  "Coventry City": { band: "#78D0F3", trim: "#FFFFFF", abbr: "COV" },
  "Crystal Palace": { band: "#1B458F", trim: "#C4122E", abbr: "CRY" },
  Everton: { band: "#003399", trim: "#FFFFFF", abbr: "EVE" },
  Fulham: { band: "#FFFFFF", trim: "#000000", abbr: "FUL" },
  "Hull City": { band: "#F5A12D", trim: "#000000", abbr: "HUL" },
  "Ipswich Town": { band: "#3A64A3", trim: "#FFFFFF", abbr: "IPS" },
  Leeds: { band: "#FFFFFF", trim: "#1D428A", abbr: "LEE" },
  Leicester: { band: "#003090", trim: "#FDBE11", abbr: "LEI" },
  Liverpool: { band: "#C8102E", trim: "#00B2A9", abbr: "LIV" },
  Luton: { band: "#F78F1E", trim: "#002D62", abbr: "LUT" },
  "Man City": { band: "#6CABDD", trim: "#1C2C5B", abbr: "MCI" },
  "Man Utd": { band: "#DA291C", trim: "#FBE122", abbr: "MUN" },
  Middlesbrough: { band: "#E21C38", trim: "#FFFFFF", abbr: "MID" },
  Newcastle: { band: "#241F20", trim: "#FFFFFF", abbr: "NEW" },
  Norwich: { band: "#FFF200", trim: "#00A650", abbr: "NOR" },
  "Nott'm Forest": { band: "#DD0000", trim: "#FFFFFF", abbr: "NFO" },
  Sheffield: { band: "#EE2737", trim: "#FFFFFF", abbr: "SHU" },
  Southampton: { band: "#D71920", trim: "#130C0E", abbr: "SOU" },
  Spurs: { band: "#FFFFFF", trim: "#132257", abbr: "TOT" },
  "Stoke City": { band: "#E03A3E", trim: "#FFFFFF", abbr: "STK" },
  Sunderland: { band: "#EB172B", trim: "#FFFFFF", abbr: "SUN" },
  Swansea: { band: "#FFFFFF", trim: "#000000", abbr: "SWA" },
  Watford: { band: "#FBEE23", trim: "#ED2127", abbr: "WAT" },
  "West Brom": { band: "#122F67", trim: "#FFFFFF", abbr: "WBA" },
  "West Ham": { band: "#7A263A", trim: "#1BB1E7", abbr: "WHU" },
  Wolves: { band: "#FDB913", trim: "#231F20", abbr: "WOL" },
};

const ALIASES: Record<string, string> = {
  "Manchester City": "Man City",
  "Manchester Utd": "Man Utd",
  "Manchester United": "Man Utd",
  "Nottingham Forest": "Nott'm Forest",
  Tottenham: "Spurs",
  "Tottenham Hotspur": "Spurs",
  "Newcastle Utd": "Newcastle",
  "Brighton & Hove Albion": "Brighton",
  "Wolverhampton": "Wolves",
  "Leeds United": "Leeds",
  "Sheffield Utd": "Sheffield",
};

const FALLBACK: ClubColours = { band: "#3D5C52", trim: "#EAF2EE", abbr: "—" };

export function clubColours(team: string): ClubColours {
  if (!team) return FALLBACK;
  const key = ALIASES[team] ?? team;
  return CLUBS[key] ?? FALLBACK;
}

export function clubAbbr(team: string, provided?: string): string {
  if (provided) return provided.toUpperCase();
  return clubColours(team).abbr;
}
