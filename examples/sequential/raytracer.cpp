// [header]
// A very basic raytracer example.
// [/header]
// [compile]
// c++ -o raytracer -O3 -Wall raytracer.cpp
// [/compile]
// [ignore]
// Copyright (C) 2012  www.scratchapixel.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// [/ignore]
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <cstring>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include "rapidxml-1.13/rapidxml.hpp"

#if defined __linux__ || defined __APPLE__
// "Compiled for Linux
#else
// Windows doesn't define these values by default, Linux does
#define M_PI 3.141592653589793
#define INFINITY 1e8
#endif

// This variable controls the maximum recursion depth
#define MAX_RAY_DEPTH 5
// This variable controls if it should work in memory. If it is not defined, works in disk
#define WORK_IN_MEMORY

#ifdef WORK_IN_MEMORY
#define WORKING_MEDIA "memory"
#else
#define WORKING_MEDIA "disk"
#endif

template<typename T>
class Vec3
{
public:
    T x, y, z;
    Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
    Vec3(T xx) : x(xx), y(xx), z(xx) {}
    Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
    Vec3& normalize()
    {
        T nor2 = length2();
        if (nor2 > 0) {
            T invNor = 1 / sqrt(nor2);
            x *= invNor, y *= invNor, z *= invNor;
        }
        return *this;
    }
    Vec3<T> operator * (const T &f) const { return Vec3<T>(x * f, y * f, z * f); }
    Vec3<T> operator * (const Vec3<T> &v) const { return Vec3<T>(x * v.x, y * v.y, z * v.z); }
    T dot(const Vec3<T> &v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3<T> operator - (const Vec3<T> &v) const { return Vec3<T>(x - v.x, y - v.y, z - v.z); }
    Vec3<T> operator + (const Vec3<T> &v) const { return Vec3<T>(x + v.x, y + v.y, z + v.z); }
    Vec3<T>& operator += (const Vec3<T> &v) { x += v.x, y += v.y, z += v.z; return *this; }
    Vec3<T>& operator *= (const Vec3<T> &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }
    Vec3<T> operator - () const { return Vec3<T>(-x, -y, -z); }
    T length2() const { return x * x + y * y + z * z; }
    T length() const { return sqrt(length2()); }
    friend std::ostream & operator << (std::ostream &os, const Vec3<T> &v)
    {
        os << "[" << v.x << " " << v.y << " " << v.z << "]";
        return os;
    }
};

typedef Vec3<float> Vec3f;
typedef Vec3<bool> Vec3b;

class Sphere
{
public:
    const char *id;
    Vec3f center;                           /// position of the sphere
    float radius, radius2;                  /// sphere radius and radius^2
    Vec3f surfaceColor, emissionColor;      /// surface color and emission (light)
    float transparency, reflection;         /// surface transparency and reflectivity
    int animation_frame;
    Vec3b animation_position_rand;
    Vec3f animation_position;
    Sphere() { }
    Sphere(
        const char *id,
        const Vec3f &c,
        const float &r,
        const Vec3f &sc,
        const float &refl = 0,
        const float &transp = 0,
        const Vec3f &ec = 0) :
        id(id), center(c), radius(r), radius2(r * r), surfaceColor(sc),
        emissionColor(ec), transparency(transp), reflection(refl)
    {
        animation_frame = 0;
    }
    //[comment]
    // Compute a ray-sphere intersection using the geometric solution
    //[/comment]
    bool intersect(const Vec3f &rayorig, const Vec3f &raydir, float &t0, float &t1) const
    {
        Vec3f l = center - rayorig;
        float tca = l.dot(raydir);
        if (tca < 0) return false;
        float d2 = l.dot(l) - tca * tca;
        if (d2 > radius2) return false;
        float thc = sqrt(radius2 - d2);
        t0 = tca - thc;
        t1 = tca + thc;
        
        return true;
    }
};

float mix(const float &a, const float &b, const float &mix)
{
    return b * mix + a * (1 - mix);
}

//[comment]
// This is the main trace function. It takes a ray as argument (defined by its origin
// and direction). We test if this ray intersects any of the geometry in the scene.
// If the ray intersects an object, we compute the intersection point, the normal
// at the intersection point, and shade this point using this information.
// Shading depends on the surface property (is it transparent, reflective, diffuse).
// The function returns a color for the ray. If the ray intersects an object that
// is the color of the object at the intersection point, otherwise it returns
// the background color.
//[/comment]
Vec3f trace(
    const Vec3f &rayorig,
    const Vec3f &raydir,
    const Sphere *spheres,
    const unsigned int spheres_size,
    const int &depth)
{
    //if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
    float tnear = INFINITY;
    const Sphere* sphere = NULL;
    // find intersection of this ray with the sphere in the scene
    for (unsigned i = 0; i < spheres_size; ++i) {
        float t0 = INFINITY, t1 = INFINITY;
        if (spheres[i].intersect(rayorig, raydir, t0, t1)) {
            if (t0 < 0) t0 = t1;
            if (t0 < tnear) {
                tnear = t0;
                sphere = &spheres[i];
            }
        }
    }
    // if there's no intersection return black or background color
    if (!sphere) return Vec3f(2);
    Vec3f surfaceColor = 0; // color of the ray/surfaceof the object intersected by the ray
    Vec3f phit = rayorig + raydir * tnear; // point of intersection
    Vec3f nhit = phit - sphere->center; // normal at the intersection point
    nhit.normalize(); // normalize normal direction
    // If the normal and the view direction are not opposite to each other
    // reverse the normal direction. That also means we are inside the sphere so set
    // the inside bool to true. Finally reverse the sign of IdotN which we want
    // positive.
    float bias = 1e-4; // add some bias to the point from which we will be tracing
    bool inside = false;
    if (raydir.dot(nhit) > 0) nhit = -nhit, inside = true;
    if ((sphere->transparency > 0 || sphere->reflection > 0) && depth < MAX_RAY_DEPTH) {
        float facingratio = -raydir.dot(nhit);
        // change the mix value to tweak the effect
        float fresneleffect = mix(pow(1 - facingratio, 3), 1, 0.1);
        // compute reflection direction (not need to normalize because all vectors
        // are already normalized)
        Vec3f refldir = raydir - nhit * 2 * raydir.dot(nhit);
        refldir.normalize();
        Vec3f reflection = trace(phit + nhit * bias, refldir, spheres, spheres_size, depth + 1);
        Vec3f refraction = 0;
        // if the sphere is also transparent compute refraction ray (transmission)
        if (sphere->transparency) {
            float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
            float cosi = -nhit.dot(raydir);
            float k = 1 - eta * eta * (1 - cosi * cosi);
            Vec3f refrdir = raydir * eta + nhit * (eta * cosi - sqrt(k));
            refrdir.normalize();
            refraction = trace(phit - nhit * bias, refrdir, spheres, spheres_size, depth + 1);
        }
        // the result is a mix of reflection and refraction (if the sphere is transparent)
        surfaceColor = (
            reflection * fresneleffect +
            refraction * (1 - fresneleffect) * sphere->transparency) * sphere->surfaceColor;
    }
    else {
        // it's a diffuse object, no need to raytrace any further
        for (unsigned i = 0; i < spheres_size; ++i) {
            if (spheres[i].emissionColor.x > 0) {
                // this is a light
                Vec3f transmission = 1;
                Vec3f lightDirection = spheres[i].center - phit;
                lightDirection.normalize();
                for (unsigned j = 0; j < spheres_size; ++j) {
                    if (i != j) {
                        float t0, t1;
                        if (spheres[j].intersect(phit + nhit * bias, lightDirection, t0, t1)) {
                            transmission = 0;
                            break;
                        }
                    }
                }
                surfaceColor += sphere->surfaceColor * transmission *
                std::max(float(0), nhit.dot(lightDirection)) * spheres[i].emissionColor;
            }
        }
    }
    
    return surfaceColor + sphere->emissionColor;
}


void save_image(const std::string output_folder, const int frame, const unsigned int width, const unsigned int height, Vec3f *image) {
    // Save result to a PPM image (keep these flags if you compile under Windows)
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(5) << frame;
    std::string filename = output_folder + "/frame" + ss.str() + ".ppm";
#ifdef DEBUG
    std::cout << "[Work] Writing frame " << frame << " to " << filename << std::endl;
#endif
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (unsigned i = 0; i < width * height; ++i) {
        ofs << (unsigned char)(std::min(float(1), image[i].x) * 255) <<
            (unsigned char)(std::min(float(1), image[i].y) * 255) <<
            (unsigned char)(std::min(float(1), image[i].z) * 255);
    }
    ofs.close();
}


void raytrace(std::string output_folder, int total_frames, unsigned int width, unsigned int height, const std::vector<Sphere> &initial_spheres) {
    float invWidth = 1 / float(width);
    float invHeight = 1 / float(height);
    float fov = 30;
    float aspectratio = width / float(height);
    float angle = tan(M_PI * 0.5 * fov / 180.);

#ifdef WORK_IN_MEMORY
    unsigned int total_memory = sizeof(Vec3f)*total_frames*width*height;
    std::string total_memory_unit = " bytes";
    if (total_memory > 1024) {
        total_memory = (total_frames*width*height)/1024;
        total_memory_unit = " KB";
    }
    if (total_memory > (10*1024)) {
        total_memory /= 1024;
        total_memory_unit = " MB";
    }
#ifdef DEBUG
    std::cout << "[Init] Allocating " << total_memory << total_memory_unit << " of memory to store images" << std::endl;
#endif
    Vec3f **images = new Vec3f*[total_frames];
    for (int f=0; f<total_frames; f++) {
        images[f] = new Vec3f[width * height];
    }
#endif

#   ifndef NO_TIME_MEASUREMENT
#ifdef DEBUG
    std::cout << "[Time] Starting time measurement" << std::endl;
#endif
    time_t wall_start, wall_end;
    time(&wall_start);
    clock_t cpu_start = clock();
#   endif

    for (int frame = 1; frame <= total_frames; frame++) {
#ifdef DEBUG
        std::cout << "[Work] Generating frame " << frame << "..." << std::endl;
#endif
        // Set up the scenne
        unsigned int spheres_size = initial_spheres.size();
        Sphere* spheres = new Sphere[spheres_size];
        memcpy(spheres, initial_spheres.data(), sizeof(Sphere) * spheres_size);

        // Animation of each frame
        for(unsigned long i = 0; i != spheres_size; i++) {
            if (spheres[i].animation_frame != 0 &&
                    (spheres[i].animation_frame > 0 && frame < spheres[i].animation_frame)) {
                continue;
            }

            int adjusted_frame = frame;
            if (spheres[i].animation_frame < 0) {
                if (frame > spheres[i].animation_frame*-1) {
                    adjusted_frame = spheres[i].animation_frame*-1;
                }
            } else if (spheres[i].animation_frame > 0) {
                adjusted_frame -= spheres[i].animation_frame;
            }

            if (spheres[i].animation_position.x) {
                if (spheres[i].animation_position_rand.x) {
                    spheres[i].center.x += (drand48()*spheres[i].animation_position.x);
                } else {
                    spheres[i].center.x += adjusted_frame*spheres[i].animation_position.x;
                }
            }
            if (spheres[i].animation_position.y) {
                if (spheres[i].animation_position_rand.y) {
                    spheres[i].center.y += (drand48()*spheres[i].animation_position.y);
                } else {
                    spheres[i].center.y += adjusted_frame*spheres[i].animation_position.y;
                }
            }
            if (spheres[i].animation_position.z) {
                if (spheres[i].animation_position_rand.z) {
                    spheres[i].center.z += (drand48()*spheres[i].animation_position.z);
                } else {
                    spheres[i].center.z += adjusted_frame*spheres[i].animation_position.z;
                }
            }
        }

#ifdef WORK_IN_MEMORY
        Vec3f *image = images[frame-1];
#else
        Vec3f *image = new Vec3f[width * height];
#endif
        // Trace rays
        for (unsigned y = 0; y < height; ++y) {
            for (unsigned x = 0; x < width; ++x) {
                float xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio;
                float yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle;
                Vec3f raydir(xx, yy, -1);
                raydir.normalize();
                image[y*width+x] = trace(Vec3f(0), raydir, spheres, spheres_size, 0);
            }
        }

#ifndef WORK_IN_MEMORY
        save_image(output_folder, frame, width, height, image);
        delete [] image;
#endif
    }

#   ifndef NO_TIME_MEASUREMENT
#ifdef DEBUG
    std::cout << "[Time] Stopping time measurement" << std::endl;
#endif
    clock_t cpu_end = clock();
    time(&wall_end);
    double cpu_time_seconds = ((double) (cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    double wall_time_seconds = difftime(wall_end, wall_start);
    printf("The generation of %d frames in %s of %u x %u with %lu spheres took:\n", total_frames, WORKING_MEDIA, width, height, initial_spheres.size());
    printf("%.0f wall-clock seconds (%.2f FPS)\n", wall_time_seconds, ((double)total_frames)/wall_time_seconds);
    printf("%.2f CPU time seconds\n", cpu_time_seconds);
#   endif

#ifdef WORK_IN_MEMORY
    for (int frame = 1; frame <= total_frames; frame++) {
        save_image(output_folder, frame, width, height, images[frame-1]);
        delete [] images[frame-1];
    }
    delete [] images;
#endif
}


int main(int argc, char **argv)
{
    int image_size_parameter = 2;
    int total_frames = 1;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <scene.xml> <output_folder>[ <image_size>[ <frames>]]" << std::endl;
        std::cerr << " <scene.xml>: XML with the scene description" << std::endl;
        std::cerr << " <output_folder>: Folder on which produce output images" << std::endl;
        std::cerr << " <image_size>: Size of images to generate, a single integer meaning 1=320x180, 2=640x360, 4=HD, 6=FHD and so on. Defaults to " << image_size_parameter << std::endl;
        std::cerr << " <frames>: Number of frames to produce. Defaults to " << total_frames << std::endl;
        exit(1);
    }
    srand48(13);

    std::string scene_filename(argv[1]);
    std::string output_folder = argv[2];
    if (argc > 3) {
        image_size_parameter = atoi(argv[3]);
    }
    if (argc > 4) {
        total_frames = atoi(argv[4]);
    }

    // 1 = 320x180
    // 2 = 640x360
    // 4 = 1280x720 (HD)
    // 6 = 1920x1080 (FHD)
    unsigned int image_size_multiplier = 20*image_size_parameter;

    unsigned int width = image_size_multiplier*16;
    unsigned int height = image_size_multiplier*9;
    
    std::vector<Sphere> initial_spheres;

#ifdef DEBUG
    std::cout << "[Init] Generating " << total_frames << " frames of " << width << "x" << height << " in " << WORKING_MEDIA << " in " << output_folder << std::endl;
    std::cout << "[Init] Loading scene from " << scene_filename << std::endl;
#endif

    // Parses the scene
    std::ifstream scene_file(scene_filename, std::ios::binary | std::ios::ate);
    std::streamsize scene_file_size = scene_file.tellg();
    scene_file.seekg(0, std::ios::beg);
    char *scene_buffer = new char[scene_file_size];
    if (scene_file.read(scene_buffer, scene_file_size)) {
        rapidxml::xml_document<> doc;
        doc.parse<0>(scene_buffer);
        rapidxml::xml_node<> *scene_node = doc.first_node("scene");

        rapidxml::xml_node<> *spheres_node = scene_node->first_node("spheres");
        rapidxml::xml_node<> *sphere_node = spheres_node->first_node();
        while (sphere_node != 0) {
            // position, radius, surface color, reflectivity, transparency, emission color
            initial_spheres.emplace_back(
                sphere_node->first_attribute("id")->value(),
                Vec3f( //Center position
                    atof(sphere_node->first_node("position")->first_attribute("x")->value()),
                    atof(sphere_node->first_node("position")->first_attribute("y")->value()),
                    atof(sphere_node->first_node("position")->first_attribute("z")->value())
                ),
                atof(sphere_node->first_node("size")->first_attribute("radius")->value()), // Radius
                Vec3f( //Surface color
                    atof(sphere_node->first_node("surface_color")->first_attribute("red")->value()),
                    atof(sphere_node->first_node("surface_color")->first_attribute("green")->value()),
                    atof(sphere_node->first_node("surface_color")->first_attribute("blue")->value())
                ),
                atof(sphere_node->first_node("reflectivity")->first_attribute("value")->value()), // Reflectivity
                atof(sphere_node->first_node("transparency")->first_attribute("value")->value()) // Transparency
            );
            if (sphere_node->first_node("emission_color")) {
                initial_spheres.back().emissionColor = Vec3f(
                    atof(sphere_node->first_node("emission_color")->first_attribute("red")->value()),
                    atof(sphere_node->first_node("emission_color")->first_attribute("green")->value()),
                    atof(sphere_node->first_node("emission_color")->first_attribute("blue")->value())
                );
            }
            sphere_node = sphere_node->next_sibling();
        }
#ifdef DEBUG
        std::cout << "[Init] Loaded " << initial_spheres.size() << " spheres, looking for animations" << std::endl;
#endif

        rapidxml::xml_node<> *animation_node = scene_node->first_node("animation");
        for (rapidxml::xml_node<> *sphere_animation = animation_node->first_node();
                sphere_animation; sphere_animation = sphere_animation->next_sibling()) {
            std::string id = sphere_animation->first_attribute("id")->value();
            for(unsigned long i = 0; i != initial_spheres.size(); i++) {
                if (id == initial_spheres[i].id) {
                    rapidxml::xml_node<> *position_node = sphere_animation->first_node("position");
                    if (position_node) {
                        rapidxml::xml_attribute<> *attr;
                        attr = position_node->first_attribute("after");
                        if (attr) {
                            initial_spheres[i].animation_frame = atoi(attr->value());
                        }
                        attr = position_node->first_attribute("before");
                        if (attr) {
                            initial_spheres[i].animation_frame = atoi(attr->value())*-1;
                        }
                        attr = position_node->first_attribute("x");
                        if (attr) {
                            if (strcmp(attr->value(), "random") == 0) {
                                initial_spheres[i].animation_position_rand.x = true;
                                initial_spheres[i].animation_position.x = atof(position_node->first_attribute("random")->value());
                            } else {
                                initial_spheres[i].animation_position.x = atof(attr->value());
                            }
                        }
                        attr = position_node->first_attribute("y");
                        if (attr) {
                            if (strcmp(attr->value(), "random") == 0) {
                                initial_spheres[i].animation_position_rand.y = true;
                                initial_spheres[i].animation_position.y = atof(position_node->first_attribute("random")->value());
                            } else {
                                initial_spheres[i].animation_position.y = atof(position_node->first_attribute("y")->value());
                            }
                        }
                        attr = position_node->first_attribute("z");
                        if (attr) {
                            if (strcmp(attr->value(), "random") == 0) {
                                initial_spheres[i].animation_position_rand.z = true;
                                initial_spheres[i].animation_position.z = atof(position_node->first_attribute("random")->value());
                            } else {
                                initial_spheres[i].animation_position.z = atof(position_node->first_attribute("z")->value());
                            }
                        }
                    }
                }
            }
        }
#ifdef DEBUG
        std::cout << "[Init] Finished loading animation for spheres" << std::endl;
#endif

    }

    raytrace(output_folder, total_frames, width, height, initial_spheres);

    return 0;
}