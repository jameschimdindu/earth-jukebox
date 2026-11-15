# Earth Jukebox

**Close your eyes. Hear the Earth.**

Earth Jukebox turns NASA's pictures of Earth into music you can understand without a screen. A voice guide called Juke talks with you, plays the data, and tells you what each sound means.

Live: https://heartheearth.vercel.app (best with headphones)

Made for the NASA Space Apps Challenge 2026, challenge "The Earth Information Jukebox", by team Lab rats (Vaasa, Finland).

## Why

NASA's Earth Information Center shows our planet as colour maps. If you cannot see well, a colour map tells you nothing. Earth Jukebox reads those maps back into numbers and plays them, so you can hear what the map shows.

## What you can do

- **Talk or type.** Juke starts at your place, or in Vaasa, Finland, if you do not share your location. Ask for any city, or drag the globe.
- **Hear a record.** Six records from the Earth Information Center: sea surface temperature, sea surface temperature anomaly, near-surface air temperature, polar sea ice, ocean plankton (global biosphere) and active fires.
- **Learn the key first.** Juke plays the usual note for the place, a higher note and a drum, and explains them in two sentences.
- **Listen in space.** A higher note means a higher value. Places to the west sound on your left, places to the east on your right.
- **Today** (key D): the newest note at a place.
- **Then and now** (key Y): the same day a year apart, the older one in your left ear and the newer one in your right.
- **Two places at once** (key K): one place in each ear.
- **Moments** (key M): the highest, the lowest and the biggest change at a place.
- **Sound postcard** (key P): saves what you hear as a WAV file, with its story and a link inside the file.
- **Share a link** (key C) that opens the same place and record.

## The data, and how exact it is

Every note comes from a value decoded from NASA's own frames (NASA Scientific Visualization Studio). The pipeline matches each pixel to the record's colour scale and keeps one value per 1-degree cell. Each record is checked against an independent source:

| Record | One note is | Checked against | Result |
|---|---|---|---|
| Sea surface temperature (SVS 5101) | one day | JPL MUR SST (NOAA ERDDAP) | 0.20 °C mean absolute error, 95% within 0.58 °C |
| Sea surface temperature anomaly (SVS 5176) | one day | MUR SST anomaly | 0.17 °C, 95% within 0.56 °C |
| Near-surface air temperature (SVS 5147) | 6 hours | NASA GEOS-FP | about 0.7 °C, more on coasts and in mountains |
| Polar sea ice (SVS 5099) | one day | MUR sea ice fraction | 5.8 percentage points; the same answer (ice or no ice) in 94% of checks |
| Ocean plankton (SVS 5075) | one week | NOAA VIIRS chlorophyll, on days not used to learn the colours | median error about 14% |
| Active fires (SVS 5113) | one week | NASA FIRMS | where the picture shows fire, FIRMS agrees in 86% of 1-degree squares |

Known limits:

- Plankton ends on 17 February 2025, where NASA stopped the record. It covers the ocean only, because NASA's frames have no plant values on land.
- Air covers about 15 days, and its last hours are NASA's forecast. Juke says so.
- The newest fire days stay empty until NASA's data arrives.

## Data updates

A GitHub Actions job runs every day at 12:00 UTC. It decodes only NASA's new frames, rebuilds the app data and publishes it. Before the air data is used, its timing is checked against NASA's GEOS-FP model.

## How it is built

- **App:** one HTML file (`mockups/voice-native.html`). three.js draws the globes. The Web Audio API plays the notes and places them in 3D.
- **Voice:** an ElevenLabs agent with Claude (Anthropic) as the language model. Juke controls the app through 20 client tools and only says numbers that the tools give it.
- **Pipeline:** Python (NumPy, SciPy, Pillow, OpenEXR) in `pipeline/`. `pipeline/rolling.py` runs the daily update.
- **Hosting:** Vercel. A small function in `api/` gives the browser a short-lived token for the voice agent, so no key reaches the browser.
- **Names of places and seas:** Open-Meteo (place search), BigDataCloud (land) and the Marine Regions gazetteer (seas).

## Accessibility

Audio comes first. Everything works with a keyboard and a screen reader: captions, a full transcript, keyboard shortcuts, and buttons for the choices of every question Juke asks.

## Run it locally

Serve the `mockups/` folder (for example `python3 -m http.server` inside it) and open `voice-native.html`. Talking to Juke needs the token function: run `vercel dev` with an `ELEVENLABS_API_KEY` that you set yourself. Each pipeline script says at the top how to run it with `uv`.

## Use of AI

Juke's voice is AI-generated (ElevenLabs), and Juke's words come from Claude. Claude also helped write the code, the data pipeline and the text.

## Credits

- **Data:** NASA Scientific Visualization Studio and the Earth Information Center (https://earth.gov), JPL MUR SST through NOAA CoastWatch ERDDAP, NASA GMAO GEOS-FP, NOAA VIIRS chlorophyll, NASA FIRMS.
- **Names:** Marine Regions (Flanders Marine Institute), BigDataCloud, Open-Meteo.
- **Libraries:** three.js (MIT), metal-fx (MIT), React (MIT), ElevenLabs client (MIT).
- **Fonts:** Geist and Geist Mono (SIL Open Font License, see `mockups/fonts/geist/LICENSE.txt`).

## Team

Lab rats, Vaasa, Finland:

- James Chimdindu Ogbonna
- Naganandhini Somalinga Mohanlal

## Licence

Our original work is licensed under the Apache License 2.0 (see `LICENSE`), the default licence in the NASA Space Apps Challenge terms. Third-party libraries and fonts keep their own licences (listed under Credits). NASA data and pictures come from NASA and are used under NASA's media guidelines.
