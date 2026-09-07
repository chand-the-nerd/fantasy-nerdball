/**
 * Case- and accent-insensitive comparison for player names.
 *
 * NFD decomposition handles most of it (é, á, ü), but a handful of Latin
 * letters are distinct characters rather than accented ones and don't
 * decompose at all. The Premier League has plenty of them, so nobody typing
 * "odegaard" or "hojlund" should come up empty.
 */
const SPECIAL: Record<string, string> = {
  ø: "o",
  æ: "ae",
  å: "a",
  œ: "oe",
  ß: "ss",
  đ: "d",
  ð: "d",
  þ: "th",
  ł: "l",
  ı: "i",
};

export function normalise(value: string): string {
  return value
    .toLowerCase()
    .replace(/[øæåœßđðþłı]/g, (char) => SPECIAL[char] ?? char)
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim();
}
