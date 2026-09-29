import qupath.lib.common.GeneralTools
import qupath.lib.objects.TMACoreObject
import qupath.lib.objects.hierarchy.TMAGrid
import qupath.lib.regions.RegionRequest

import java.awt.Color
import java.awt.image.BufferedImage
import javax.imageio.ImageIO

import static qupath.lib.gui.scripting.QPEx.*

if (args.length != 1)
    throw new IllegalArgumentException("Usage: export_cores_qupath.groovy MORPHOLOGY_DATA_DIRECTORY")

double downsample = 1.0
def imageData = getCurrentImageData()
if (imageData == null)
    throw new IllegalStateException("No image is open")

def server = imageData.getServer()
def calibration = server.getPixelCalibration()
double pixelWidthMicrons = calibration.getPixelWidthMicrons()
double pixelHeightMicrons = calibration.getPixelHeightMicrons()
if (!Double.isFinite(pixelWidthMicrons) || !Double.isFinite(pixelHeightMicrons))
    throw new IllegalStateException("The image has no physical pixel calibration")
if (pixelWidthMicrons > 0.6 || pixelHeightMicrons > 0.6 || Math.max(server.getWidth(), server.getHeight()) < 50000)
    throw new IllegalStateException("Open the full-resolution 20x_BF_01 series before exporting")

TMAGrid grid = imageData.getHierarchy().getTMAGrid()
if (grid == null || grid.getGridHeight() != 5 || grid.getGridWidth() != 6)
    throw new IllegalStateException("A 5 x 6 TMA grid is required")

String imageName = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())
def matcher = imageName =~ /TMA LIP-(\d+) (6E10|AT8|NeuN)/
if (!matcher.find())
    throw new IllegalStateException("Could not identify the LIP array and stain from: ${imageName}")
String tma = matcher.group(1)
String stain = matcher.group(2)

int exported = 0
for (int row = 0; row < grid.getGridHeight(); row++) {
    for (int column = 0; column < grid.getGridWidth(); column++) {
        TMACoreObject core = grid.getTMACore(row, column)
        if (core == null || core.getROI() == null)
            throw new IllegalStateException("Missing ROI at row ${row + 1}, column ${column + 1}")

        def roi = core.getROI()
        String label = core.getName()
        if (label == null || label.trim().isEmpty())
            label = String.format("%c-%d", (char)(65 + column), row + 1)
        String safeLabel = label.replaceAll(/[^A-Za-z0-9_\-]/, "_")
        File outputDirectory = new File(args[0], "cores/LIP-${tma}/${safeLabel}")
        outputDirectory.mkdirs()
        File outputPath = new File(outputDirectory, "${stain}.png")

        def request = RegionRequest.createInstance(server.getPath(), downsample, roi)
        int clipX = Math.max(0, request.getX())
        int clipY = Math.max(0, request.getY())
        int clipMaxX = Math.min(server.getWidth(), request.getX() + request.getWidth())
        int clipMaxY = Math.min(server.getHeight(), request.getY() + request.getHeight())
        if (clipMaxX <= clipX || clipMaxY <= clipY)
            throw new IllegalStateException("Core ${label} does not intersect the image")
        def clippedRequest = RegionRequest.createInstance(
            server.getPath(),
            downsample,
            clipX,
            clipY,
            clipMaxX - clipX,
            clipMaxY - clipY,
            request.getZ(),
            request.getT()
        )
        def patch = server.readRegion(clippedRequest)
        int outputWidth = request.getWidth()
        int outputHeight = request.getHeight()
        def output = new BufferedImage(outputWidth, outputHeight, BufferedImage.TYPE_INT_RGB)
        def graphics = output.createGraphics()
        graphics.setColor(Color.WHITE)
        graphics.fillRect(0, 0, outputWidth, outputHeight)
        graphics.drawImage(patch, clipX - request.getX(), clipY - request.getY(), null)
        graphics.dispose()
        if (!ImageIO.write(output, "png", outputPath))
            throw new IOException("No PNG writer is available")
        exported++
    }
}

println "Exported ${exported} native-resolution positions from ${imageName}"
