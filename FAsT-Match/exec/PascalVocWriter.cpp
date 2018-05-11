//
// Created by rokas on 18.5.1.
//

#include <boost/property_tree/xml_parser.hpp>
#include "PascalVocWriter.hpp"

PascalVocWriter::PascalVocWriter(const fs::path& imagePath, const cv::Size& imsize) {
    fs::path xmlPath = imagePath;
    xmlPath.replace_extension(".xml");
    outfilename = xmlPath.string();
     //truksta dar verified <annotation verified="yes">
    tree.put("annotation.folder", imagePath.root_directory().string());
    tree.put("annotation.filename", imagePath.filename().string());
    tree.put("annotation.path", outfilename);
    tree.put("annotation.source.database", "aerial");
    tree.put("annotation.size.width", imsize.width);
    tree.put("annotation.size.height", imsize.height);
    tree.put("annotation.size.depth","3");
    tree.put("annotation.segmented","1");
}

void PascalVocWriter::addPolygon(const std::vector<cv::Point> &poly, const std::string& name) {
    unsigned long pointsNumb = poly.size(); //std::string -> int
    std::string ob = "object.bndbox.";
    pt::ptree subtree;
    subtree.add("object", "");
    subtree.add("object.name", name);
    subtree.add("object.pose", "Unspecified");
    subtree.add("object.difficult", "0");//bool
    subtree.add("object.tetragon", "0");
    subtree.add("object.shape3D", "0");
    subtree.add("object.polygon", "1");
    //subtree.add("object.polygons", each_object["polygons"]);
    subtree.add("object.angle", "0");
    subtree.add("object.truncated","0");
    for(int i = 0; i < pointsNumb; i++) {
        std::string kx = "k" + std::to_string(i) + "x";
        std::string ky = "k" + std::to_string(i) + "y";
        subtree.add(ob + kx, poly[i].x);
        subtree.add(ob + ky, poly[i].y);
    }
    tree.add_child("annotation.object", subtree.get_child("object"));
}

void PascalVocWriter::write() const {
    pt::write_xml(outfilename, tree);
}
