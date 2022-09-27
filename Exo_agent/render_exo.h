#include "/home/medicalrobotics/MASS_EXO/render/Window.h"
#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
namespace py = pybind11;
namespace MASS
{
// class Environment;
// class Muscle;
class exo_Window : public Window 
{
public:
    exo_Window(Environment*, const std::string&, const std::string&, const std::string&);
    void Step()override;
    Eigen::VectorXd GetExoTorquesFromNN();

    py::object exo_agent;
};
}