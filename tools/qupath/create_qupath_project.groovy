import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.projects.Projects

import java.awt.image.BufferedImage

if (args.length != 2)
    throw new IllegalArgumentException("Usage: create_project.groovy SLIDES_DIRECTORY PROJECT_FILE")

def slidesDirectory = new File(args[0])
def projectFile = new File(args[1])
if (!slidesDirectory.isDirectory())
    throw new IllegalArgumentException("Slides directory does not exist: ${slidesDirectory}")
if (projectFile.exists())
    throw new IllegalStateException("Project already exists: ${projectFile}")
projectFile.getParentFile().mkdirs()

def slides = slidesDirectory.listFiles()
    .findAll { file -> file.isFile() && file.getName().toLowerCase().endsWith(".vsi") }
    .sort { file -> file.getName() }
if (slides.size() != 21)
    throw new IllegalStateException("Expected 21 VSI slides, found ${slides.size()}")

def project = Projects.createProject(projectFile, BufferedImage.class)
slides.each { slide ->
    def support = ImageServerProvider.getPreferredUriImageSupport(
        BufferedImage.class,
        slide.getAbsolutePath(),
        "--classname",
        "BioFormatsServerBuilder",
        "--series",
        "2"
    )
    def builders = support.getBuilders()
    if (builders.size() != 1)
        throw new IllegalStateException("Expected one full-resolution server for ${slide}, found ${builders.size()}")
    def entry = project.addImage(builders[0])
    entry.setImageName(slide.getName().replaceFirst(/(?i)\.vsi$/, ""))
}
project.syncChanges()

println "Created ${project.getPath()} with ${project.getImageList().size()} images"
