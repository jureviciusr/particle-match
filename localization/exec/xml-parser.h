/***************************************************************************
 *  Project:    osm-router
 *  File:       xml-parser.h
 *  Language:   C++
 *
 *  osm-router is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by the Free
 *  Software Foundation, either version 3 of the License, or (at your option)
 *  any later version.
 *
 *  osm-router is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 *  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 *  more details.
 *
 *  You should have received a copy of the GNU General Public License along with
 *  osm-router.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Author:     Mark Padgham
 *  E-Mail:     mark.padgham@email.com
 *
 *  Description:    Extracts an OSM XML file from the overpass API and extracts
 *                  all highways and associated nodes.
 *
 *  Limitations:
 *
 *  Dependencies:       libboost
 *
 *  Compiler Options:   -std=c++11 -lboost_program_options
 ***************************************************************************/


#include <curl/curl.h>

#include <iostream>
#include <vector>
#include <string>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/unordered_map.hpp>

#ifndef XML_H
#define XML_H

typedef std::pair <float, float> ffPair; // lat-lon

typedef boost::unordered_map <long long, ffPair> umapPair;
typedef boost::unordered_map <long long, ffPair>::iterator umapPair_Itr;

// See http://theboostcpplibraries.com/boost.unordered
std::size_t hash_value(const ffPair &f)
{
    std::size_t seed = 0;
    boost::hash_combine(seed, f.first);
    boost::hash_combine(seed, f.second);
    return seed;
}

struct Node
{
    long long id;
    float lat, lon;
};

/* Traversing the boost::property_tree means keys and values are read
 * sequentially and cannot be processed simultaneously. Each way is thus
 * initially read as a RawWay with separate vectors for keys and values. These
 * are subsequently converted in Way to a vector of <std::pair>. */
struct RawWay
{
    long long id;
    std::vector <std::string> key, value;
    std::vector <long long> nodes;
};

struct Way
{
    bool oneway;
    long long id;
    std::string type, name; // type is highway type (value for highway key)
    std::vector <std::pair <std::string, std::string> > key_val;
    std::vector <long long> nodes;
};

typedef std::vector <Way> Ways;


/************************************************************************
 ************************************************************************
 **                                                                    **
 **                         CLASS::CURLPLUPLUS                         **
 **                                                                    **
 ************************************************************************
 ************************************************************************/

class CURLplusplus
{
private:
    CURL* curl;
    std::stringstream ss;
    long http_code;
public:
    CURLplusplus()
            : curl(curl_easy_init())
            , http_code(0)
    {
    }
    ~CURLplusplus()
    {
        if (curl) curl_easy_cleanup(curl);
    }
    std::string Get(const std::string& url)
    {
        CURLcode res;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);

        ss.str("");
        http_code = 0;
        res = curl_easy_perform(curl);
        if (res != CURLE_OK)
        {
            throw std::runtime_error(curl_easy_strerror(res));
        }
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        return ss.str();
    }
    long GetHttpCode()
    {
        return http_code;
    }
private:
    static size_t write_data(void *buffer, size_t size, size_t nmemb, void *userp)
    {
        return static_cast<CURLplusplus*>(userp)->Write(buffer,size,nmemb);
    }
    size_t Write(void *buffer, size_t size, size_t nmemb)
    {
        ss.write((const char*)buffer,size*nmemb);
        return size*nmemb;
    }
};


/************************************************************************
 ************************************************************************
 **                                                                    **
 **                             CLASS::XML                             **
 **                                                                    **
 ************************************************************************
 ************************************************************************/

class Xml
{
    /*
     * Downloads OSM data from the overpass API and parses the XML structure to
     * extract all nodes and ways.
     */
private:
    const std::string _file;
    float _latmin, _lonmin, _latmax, _lonmax;

protected:
    bool file_exists;

public:
    int tempi;
    std::string tempstr;
    Ways ways;
    umapPair nodes;

    Xml (std::string file, float lonmin, float latmin, float lonmax, float latmax)
            : _file (file), _lonmin (lonmin), _latmin (latmin),
              _lonmax (lonmax), _latmax (latmax)
    {
        ways.resize (0);
        nodes.clear ();

        boost::filesystem::path p (_file);
        if (boost::filesystem::exists (p))
        {
            std::ifstream in_file;
            in_file.open (_file.c_str (), std::ifstream::in);
            assert (!in_file.fail ());
            std::stringstream ss;
            ss << in_file.rdbuf ();
            tempstr = ss.str ();
        } else
        {
            std::cout << "Downloading overpass query ... ";
            std::cout.flush ();
            tempstr = readOverpass ();
            // Write raw xml data to _file:
            std::ofstream out_file;
            out_file.open (_file.c_str (), std::ofstream::out);
            out_file << tempstr;
            out_file.flush ();
            out_file.close ();
            std::cout << " stored in " << _file << std::endl;
        }

        parseXML (tempstr);
    }

    Xml();

    ~Xml ()
    {
        ways.resize (0);
        nodes.clear ();
    }

    std::string get_file () { return _file; }
    float get_lonmin () { return _lonmin;   }
    float get_latmin () { return _latmin;   }
    float get_lonmax () { return _lonmax;   }
    float get_latmax () { return _latmax;   }

    std::string readOverpass ();
    void parseXML ( std::string & is );
    void traverseXML (const boost::property_tree::ptree& pt);
    RawWay traverseWay (const boost::property_tree::ptree& pt, RawWay rway);
    Node traverseNode (const boost::property_tree::ptree& pt, Node node);
};



/************************************************************************
 ************************************************************************
 **                                                                    **
 **                       FUNCTION::READOVERPASS                       **
 **                                                                    **
 ************************************************************************
 ************************************************************************/

std::string Xml::readOverpass ()
{
    const std::string key = "['highway']",
            url_base = "http://overpass-api.de/api/interpreter?data=";
    std::stringstream bbox, query, url;

    bbox << "";
    bbox << "(" << get_latmin () << "," << get_lonmin () << "," <<
         get_latmax () << "," << get_lonmax () << ")";

    query << "";
    query << "(node" << key << bbox.str() << ";way" << key << bbox.str() <<
          ";rel" << key << bbox.str() << ";";
    url << "";
    url << url_base << query.str() << ");(._;>;);out;";

    CURLplusplus client;
    std::string x = client.Get (url.str().c_str ());

    return x;
}; // end function readOverpass


/************************************************************************
 ************************************************************************
 **                                                                    **
 **                         FUNCTION::PARSEXML                         **
 **                                                                    **
 ************************************************************************
 ************************************************************************/

void Xml::parseXML ( std::string & is )
{
    // populate tree structure pt
    using boost::property_tree::ptree;
    ptree pt;
    std::stringstream istream (is, std::stringstream::in);
    read_xml (istream, pt);

    ptree bounds = pt.get_child("osm.bounds");
    _latmin = bounds.get<float>("<xmlattr>.minlat");
    _latmax = bounds.get<float>("<xmlattr>.maxlat");
    _lonmin = bounds.get<float>("<xmlattr>.minlon");
    _lonmax = bounds.get<float>("<xmlattr>.maxlon");

    // std::cout << is << std::endl; // The overpass XML data
    traverseXML (pt);
}


/************************************************************************
 ************************************************************************
 **                                                                    **
 **                        FUNCTION::TRAVERSEXML                       **
 **                                                                    **
 ************************************************************************
 ************************************************************************/

void Xml::traverseXML (const boost::property_tree::ptree& pt)
{
    int tempi;
    RawWay rway;
    Way way;
    Node node;
    // NOTE: Node is (lon, lat) = (x, y)!

    for (boost::property_tree::ptree::const_iterator it = pt.begin ();
         it != pt.end (); ++it)
    {
        if (it->first == "node")
        {
            node = traverseNode (it->second, node);
            nodes [node.id] = std::make_pair (node.lon, node.lat);

            // ------------ just text output guff ---------------
            /*
            std::cout << "-----Node ID = " << node.id << " (" <<
                node.lon << ", " << node.lat << ")" << std::endl;
            */
        }
        if (it->first == "way")
        {
            rway.key.resize (0);
            rway.value.resize (0);
            rway.nodes.resize (0);

            rway = traverseWay (it->second, rway);
            assert (rway.key.size () == rway.value.size ());

            // This is much easier as explicit loop than with an iterator
            way.id = rway.id;
            way.name = way.type = "";
            way.key_val.resize (0);
            way.oneway = false;
            // TODO: oneway also exists is pairs:
            // k='oneway' v='yes'
            // k='oneway:bicycle' v='no'
            for (int i=0; i<rway.key.size (); i++)
                if (rway.key [i] == "name")
                    way.name = rway.value [i];
                else if (rway.key [i] == "highway")
                    way.type = rway.value [i];
                else if (rway.key [i] == "oneway" && rway.value [i] == "yes")
                    way.oneway = true;
                else
                    way.key_val.push_back (std::make_pair (rway.key [i], rway.value [i]));

            // Then copy nodes from rway to way.
            way.nodes.resize (0);
            for (std::vector <long long>::iterator it = rway.nodes.begin ();
                 it != rway.nodes.end (); it++)
                way.nodes.push_back (*it);
            ways.push_back (way);

            // ------------ just text output guff ---------------
            /*
            std::cout << "-----Way ID = " << way.id << std::endl;
            std::cout << "nodes = (";
            for (std::vector <long long>::iterator wn = way.nodes.begin ();
                    wn != way.nodes.end (); wn++)
                std::cout << *wn << ", ";
            std::cout << ")" << std::endl;
            std::cout << "NAME = " << way.name << "; type = " <<
                way.type << std::endl;
            for (std::vector <std::pair <std::string, std::string>>::iterator
                    it = way.key_val.begin (); it != way.key_val.end (); it++)
            {
                std::cout << (*it).first << ": " << (*it).second << std::endl;
            }
            */
        } else
            traverseXML (it->second);
    }
    rway.key.resize (0);
    rway.value.resize (0);
    rway.nodes.resize (0);
} // end function Xml::traverseXML

/************************************************************************
 ************************************************************************
 **                                                                    **
 **                        FUNCTION::TRAVERSEWAY                       **
 **                                                                    **
 ************************************************************************
 ************************************************************************/

RawWay Xml::traverseWay (const boost::property_tree::ptree& pt, RawWay rway)
{
    for (boost::property_tree::ptree::const_iterator it = pt.begin ();
         it != pt.end (); ++it)
    {
        if (it->first == "k")
            rway.key.push_back (it->second.get_value <std::string> ());
        else if (it->first == "v")
            rway.value.push_back (it->second.get_value <std::string> ());
        else if (it->first == "id")
            rway.id = it->second.get_value <long long> ();
        else if (it->first == "ref")
            rway.nodes.push_back (it->second.get_value <long long> ());
        rway = traverseWay (it->second, rway);
    }

    return rway;
} // end function Xml::traverseWay


/************************************************************************
 ************************************************************************
 **                                                                    **
 **                       FUNCTION::TRAVERSENODE                       **
 **                                                                    **
 ************************************************************************
 ************************************************************************/

Node Xml::traverseNode (const boost::property_tree::ptree& pt, Node node)
{
    // Only coordinates of nodes are read, because only those are stored in the
    // unordered map. More node info is unlikely to be necessary ... ?
    for (boost::property_tree::ptree::const_iterator it = pt.begin ();
         it != pt.end (); ++it)
    {
        if (it->first == "id")
            node.id = it->second.get_value <long long> ();
        else if (it->first == "lat")
            node.lat = it->second.get_value <float> ();
        else if (it->first == "lon")
            node.lon = it->second.get_value <float> ();
        // No other key-value pairs currently extracted for nodes
        node = traverseNode (it->second, node);
    }

    return node;
}

Xml::Xml() : _latmax(0.f), _latmin(0.f), _lonmax(0.f), _lonmin(0.f) {
    ways.resize (0);
    nodes.clear ();
}
// end function Xml::traverseNode

#endif