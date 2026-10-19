// Practice content.
// Words avoid J and Z (motion letters, not supported) and T (the dataset's T isn't standard ASL,
// see RESULTS.md), and don't mix P and Q (the model's most confused pair on the webcam).

export const LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY".split("");

// Letters shown with a warning in the app
export const LETTER_NOTES = {
  T: "Dataset version, not standard ASL. A real deployment needs T re-recorded with the correct handshape.",
};

export const WORDS = [
  "HELLO", "LOVE", "HAND", "SIGN", "CAFE", "BOOK", "GOOD", "FOOD", "MILK", "BLUE",
  "RED", "FISH", "BIRD", "HOME", "NAME", "FRIEND", "FAMILY", "WORLD", "HAPPY", "SMILE",
  "MOUSE", "APPLE", "BREAD", "CLOUD", "DANCE", "MUSIC", "PIANO", "SNOW", "RAIN", "FIRE",
  "OCEAN", "SUN", "MOON", "KING", "QUEEN", "HORSE", "CANDLE", "PENCIL", "CHAIR", "DOG",
  "RAINBOW", "YELLOW", "PURPLE", "ORANGE", "GREEN", "BLACK", "COFFEE", "SCHOOL", "LEARN", "SPELL",
];

const shuffle = (arr) => {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};

/** An endless shuffled supply of practice items: words, or single letters ("letters" mode). */
export function makeQueue(mode) {
  return shuffle(mode === "letters" ? LETTERS : WORDS);
}
