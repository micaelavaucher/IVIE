# Testers Playthroughs

This folder contains the playthrough recordings and form responses from the user testing sessions conducted for **IVIE** (Incrementally Validated Interactive Experiences), an AI-powered text adventure world generator.

A total of **8 testers** participated. All tester identities have been anonymized — nicknames used during gameplay have been replaced with `tester_1` through `tester_8`.

---

## Folder Structure

```
testers_playthroughs/
  tester_1/
    generated_mode.html
    inspiration_mode.html
  tester_2/
    ...
  ...
  tester_8/
    generated_mode.html
    inspiration_mode.html
  testers_form_answers.csv
```

Each tester folder contains two HTML files, one per game mode:

- **`generated_mode.html`** — playthrough where the world was generated randomly (no user-provided inspiration).
- **`inspiration_mode.html`** — playthrough where the tester provided a creative prompt to inspire the world generation.

---

## HTML Playthroughs

Each HTML file is an exported chat log of a single game session. It contains:

- **Metadata** — World ID, tester nickname (anonymized), language used (English/Spanish), inspiration text (if any), and creation date.
- **Initial Narration** — The opening description of the world presented to the player at the start of the session.
- **Conversation turns** — Each turn shows the player's input and the AI narrator's response, labeled by turn number and timestamp.

The game content (narrations, player inputs, world names, NPC dialogue) may be in Spanish or English depending on the language the tester chose to play in.

---

## Form Answers (`testers_form_answers.csv`)

After each session, testers filled out a Google Form evaluating their experience. The CSV contains only the **Likert scale questions** — open-ended comments have been excluded.

Questions are grouped into two sections matching the two game modes:

### [Generated] — Evaluating the randomly generated world

| Column | Question |
|--------|----------|
| [Generated] The puzzles made sense and had logic. | Were the puzzles coherent and logical? |
| [Generated] I was able to solve the puzzles with the world elements. | Could puzzles be solved using in-world elements? |
| [Generated] The hints were clear and useful. | Were the hints helpful? |
| [Generated] The world objective was clear from the start. | Was the goal obvious from the beginning? |
| [Generated] I was able to complete the world objective. | Did the tester finish the objective? |
| [Generated] The objects and NPCs had a real purpose. | Did items and characters feel meaningful? |
| [Generated] The game was entertaining and kept my interest. | Overall engagement with the generated world. |

### [Inspiration] — Evaluating the inspiration-based world

| Column | Question |
|--------|----------|
| [Inspiration] The puzzles made sense and had logic. | Were the puzzles coherent and logical? |
| [Inspiration] The hints were clear and useful. | Were the hints helpful? |
| [Inspiration] The objective correlated with the given inspiration. | Did the goal relate to the inspiration provided? |
| [Inspiration] The world objective was clear from the start. | Was the goal obvious from the beginning? |
| [Inspiration] I was able to complete the world objective. | Did the tester finish the objective? |
| [Inspiration] All locations were related to the given inspiration. | Did locations reflect the inspiration? |
| [Inspiration] All NPCs were related to the given inspiration. | Did characters reflect the inspiration? |
| [Inspiration] The objects and NPCs had a real purpose. | Did items and characters feel meaningful? |
| [Inspiration] The generated world and story clearly reflected the given inspiration. | Overall faithfulness to the inspiration. |
| [Inspiration] The game was entertaining and kept my interest. | Overall engagement with the inspiration-based world. |

### Likert Scale

All answers use a 5-point scale:

| Value | Meaning |
|-------|---------|
| Yes | Always / completely |
| Mostly | Most of the time |
| Sometimes | About half the time |
| Almost Never | Rarely |
| Never | Not at all |
