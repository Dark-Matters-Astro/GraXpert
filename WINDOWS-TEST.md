# Windows validation of the sample-free ONNX implementation

This procedure validates the new sample-free method from a source checkout. It
does not build a distributable GraXpert executable. The test is deliberately
isolated from an existing GraXpert installation and its preferences.

## Requirements

- Windows 10 or 11, 64-bit
- Python 3.10, 64-bit, from python.org (including Tcl/Tk)
- Git and internet access for the initial dependency installation
- 16 GB RAM recommended; testing on 8 GB is useful but may reach the memory
  limit with large images

## Set up the test checkout

Clone the repository and switch to the prepared branch:

```bat
git clone <repository-url>
cd GraXpert
git switch feature/automatic-bge
tools\setup_windows_test.cmd
```

The setup script creates `.venv-windows`, installs the dependencies and
verifies that ONNX Runtime can load the checked-in filtering graph. No GraXpert
AI model download is needed for the sample-free method.

Transfer the test FITS file separately. Do not add astronomical test images to
the repository.

## Manual GUI test

Start the isolated source version:

```bat
tools\start_windows_test.cmd
```

Then:

1. Load the colour FITS image.
2. Select `Sample-free` and keep the default parameters for the first run.
3. Start the calculation and check that the interface remains responsive and
   the calculation finishes without an error.
4. Inspect `Original`, `Background`, `Corrected` and `Simplified Model`.
5. Repeat the calculation at least five times and note the processing times.
6. Change scale, smoothness, structure protection and polynomial degree once,
   then save a corrected FITS result for visual comparison with Siril.

Application data and the log stay below `.local-test-data`. The log file is:

```text
.local-test-data\logs\graxpert.log
```

## Automated stress test

Run this from the repository directory, or drag the FITS file onto
`tools\run_windows_stress_test.cmd`:

```bat
tools\run_windows_stress_test.cmd "C:\path\to\NGC7331-L_drizzle_1x_autocrop.fit"
```

The test performs ten full colour-image calculations, repeats a parameter
matrix, checks mono and odd-sized inputs, compares ONNX with the NumPy reference
and exercises progress updates through a real Tk window. It writes:

```text
.local-test-data\onnx-windows-stress-report.json
```

All hashes for repeated runs of the same case on that Windows system must be
identical. Exact hashes need not match macOS, because floating-point execution
order can differ slightly between platforms.

## Information to report back

- Windows version, processor and installed RAM
- output of `py -3.10 --version`
- whether setup, GUI test and stress test passed
- approximate full-image processing time
- the JSON stress-test report and `graxpert.log` if there is a failure
- a short visual comparison with the Siril result

## Current reference from macOS testing

On the current Apple-silicon development machine, 20 consecutive full RGB runs
were deterministic and took roughly 7.2 to 8.3 seconds each. The complete Python
test suite passed. ONNX Runtime uses its CPU provider intentionally; the Core ML
provider was slower for this small deterministic graph.

This does not replace Windows validation. The packaged Windows executable and
Intel-specific performance have not yet been tested, and full-resolution images
with downsample factor 1 can require substantially more memory.

The method is adapted from Siril. Attribution and license details are recorded
in `licenses/SIRIL_SAMPLE_FREE_NOTICE.md`.
