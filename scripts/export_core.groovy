/**
 * Export each TMA core as an individual PNG image, using the internal TMA grid.
 *
 * - Uses the TMAGrid / TMACoreObject from the current image hierarchy
 * - Skips cores marked as 'missing'
 * - Filenames include:
 *       image name
 *       numeric column & row indices (1-based, zero-padded)
 *       the core label (e.g. A-1) if available
 *
 * Requirements:
 *   - Run inside a QuPath project (uses PROJECT_BASE_DIR)
 *   - TMA must already be dearrayed for the current image
 */

import qupath.lib.common.GeneralTools
import qupath.lib.regions.RegionRequest
import qupath.lib.objects.TMACoreObject
import qupath.lib.objects.hierarchy.TMAGrid

import static qupath.lib.gui.scripting.QPEx.*

// ----------------- User options -----------------
double downsample = 1.0          // 1.0 = full resolution, increase for smaller exports
String imageFormat = ".png"      // changed from ".tif" to ".png"
// ------------------------------------------------

// Get current image data & server
def imageData = getCurrentImageData()
if (imageData == null) {
    print "No image open!"
    return
}
def server = imageData.getServer()
def hierarchy = imageData.getHierarchy()

// Get TMA grid
TMAGrid tmaGrid = hierarchy.getTMAGrid()
if (tmaGrid == null) {
    print "No TMA grid found – has this slide been dearrayed?"
    return
}

// Get list of all cores
def cores = tmaGrid.getTMACoreList()
if (cores == null || cores.isEmpty()) {
    print "No TMA cores found in the grid."
    return
}

// Prepare output directory (requires a project)
def imageName = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())
def exportDir = buildFilePath(PROJECT_BASE_DIR, "TMA_core_exports", imageName)
mkdirs(exportDir)

print "Exporting ${cores.size()} TMA cores from image '${imageName}'"
print "Output directory: ${exportDir}"
print "Downsample: ${downsample}, Format: ${imageFormat}"

// Helper: find row & column indices for a given core
int gridHeight = tmaGrid.getGridHeight()
int gridWidth  = tmaGrid.getGridWidth()

def getRowColForCore = { TMACoreObject core ->
    int rowIndex = -1
    int colIndex = -1
    outer:
    for (int r = 0; r < gridHeight; r++) {
        for (int c = 0; c < gridWidth; c++) {
            def gridCore = tmaGrid.getTMACore(r, c)
            if (gridCore == core) {
                rowIndex = r
                colIndex = c
                break outer
            }
        }
    }
    return [rowIndex, colIndex]
}

int count = 0

cores.each { TMACoreObject core ->

    if (core.isMissing()) {
        return
    }

    def roi = core.getROI()
    if (roi == null) {
        return
    }

    def request = RegionRequest.createInstance(server.getPath(), downsample, roi)

    def (rowIndex, colIndex) = getRowColForCore(core)
    String rowStr = (rowIndex >= 0) ? String.format("%02d", rowIndex + 1) : "NA"
    String colStr = (colIndex >= 0) ? String.format("%02d", colIndex + 1) : "NA"

    String coreLabel = core.getName()
    if (coreLabel == null || coreLabel.trim().isEmpty()) {
        coreLabel = String.format("core_%03d", count + 1)
    }
    coreLabel = coreLabel.replaceAll(/[^A-Za-z0-9_\-]/, "_")

    String filename = String.format(
            "%s_col%s_row%s_%s%s",
            imageName,
            colStr,
            rowStr,
            coreLabel,
            imageFormat
    )

    def outputPath = buildFilePath(exportDir, filename)

    try {
        writeImageRegion(server, request, outputPath)
        count++
        if (count % 10 == 0)
            print "Exported ${count} cores..."
    } catch (Exception e) {
        print "Failed to export core '${coreLabel}' (col ${colStr}, row ${rowStr}): ${e.message}"
    }
}

print "Done! Exported ${count} cores to ${exportDir}"