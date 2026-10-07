# Mobile capture and preview flow

This Expo development-build app now provides the complete phone-side flow:

- a choice between an object scan (capped at 1 minute, cropped with a box) and a
  room/scene scan for rooms or several objects (capped at 5 minutes, kept uncut with an
  optional ceiling slice), each with its own preparation guidance;
- camera-based, muted 1080p recording;
- replay, retake, streaming SHA-256 calculation and direct signed uploads;
- reconstruction status polling with automatic preview navigation;
- signed GLB point-cloud preview with touch orbit and pinch/button zoom;
- 90-degree upright correction using a proper rigid transform;
- bounded min/max crop controls for all three reconstruction axes;
- required target axis and block dimension;
- server-authoritative block-count preflight and large-export warning.

The scale controls remain locked until the current crop and orientation have been
saved. Any subsequent edit invalidates that saved selection. This prevents an
estimate from being shown against stale geometry.

## Run

Start the backend with a storage URL that the phone can reach:

```bash
export VTM_S3_PUBLIC_ENDPOINT_URL=http://192.168.1.20:9000
docker compose -f infra/compose.yaml up --build
```

Then install and rebuild the development client, providing the phone-reachable API
address when Metro starts:

```bash
cd apps/mobile
pnpm install
EXPO_PUBLIC_API_URL=http://192.168.1.20:8000 pnpm android
```

Replace `192.168.1.20` with the computer's actual LAN address. For an Android
emulator, use `10.0.2.2` for both endpoints. The API and GPU reconstruction
worker must both be healthy for a new scan to finish.

On later runs, when the native dependencies have not changed, start Metro with:

```bash
EXPO_PUBLIC_API_URL=http://192.168.1.20:8000 pnpm start
```

The home screen also accepts an existing scan UUID as a recovery path. Direct
links to `/preview/<scan UUID>` continue to work.

## Verify

```bash
pnpm typecheck
pnpm test
```
