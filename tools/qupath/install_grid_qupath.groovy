import qupath.lib.objects.PathObjects
import qupath.lib.objects.TMACoreObject
import qupath.lib.objects.hierarchy.DefaultTMAGrid

import static qupath.lib.gui.scripting.QPEx.*

if (args.length != 1)
    throw new IllegalArgumentException("Usage: install_grid_qupath.groovy CORE_MANIFEST")

def imageData = getCurrentImageData()
if (imageData == null)
    throw new IllegalStateException("No image is open")

String imageName = imageData.getServer().getMetadata().getName().replaceFirst(/(?i)\.vsi$/, "")
File manifest = new File(args[0])
if (!manifest.isFile())
    throw new IllegalStateException("Core manifest does not exist: ${manifest}")

def lines = manifest.readLines("UTF-8")
def header = lines[0].split(",", -1).toList()
def required = [
    "slide", "tma", "stain", "row", "column", "core_label", "center_x_px", "center_y_px",
    "diameter_x_px", "diameter_y_px", "provisional_tissue_fraction",
    "provisional_tissue_status", "grid_rmse_preview_px"
]
required.each { column ->
    if (!header.contains(column))
        throw new IllegalStateException("Missing column '${column}' in ${manifest}")
}

def records = lines.tail().collect { line ->
    def values = line.split(",", -1)
    if (values.length != header.size())
        throw new IllegalStateException("Malformed row in ${manifest}: ${line}")
    return [header, values].transpose().collectEntries()
}.findAll { record ->
    record.slide == imageName
}.sort { record ->
    [record.row as int, record.column as int]
}
if (records.size() != 30)
    throw new IllegalStateException("Expected 30 positions for ${imageName}, found ${records.size()}")

List<TMACoreObject> cores = records.collect { record ->
    double centerX = record.center_x_px as double
    double centerY = record.center_y_px as double
    double width = record.diameter_x_px as double
    double height = record.diameter_y_px as double
    def core = PathObjects.createTMACoreObject(
        centerX - width / 2,
        centerY - height / 2,
        width,
        height,
        false
    )
    core.setName(record.core_label)
    core.putMetadataValue("TMA", "LIP-${record.tma}")
    core.putMetadataValue("Stain", record.stain)
    core.putMetadataValue("Provisional tissue status", record.provisional_tissue_status)
    core.putMetadataValue("Grid source", manifest.getAbsolutePath())
    core.getMeasurementList().put(
        "Provisional tissue fraction",
        record.provisional_tissue_fraction as double
    )
    core.getMeasurementList().put(
        "Grid fit RMSE (preview px)",
        record.grid_rmse_preview_px as double
    )
    return core
}

def hierarchy = imageData.getHierarchy()
hierarchy.setTMAGrid(DefaultTMAGrid.create(cores, 6))
hierarchy.fireHierarchyChangedEvent(this)

println "Installed 5 x 6 grid for ${imageName} from ${manifest}"
