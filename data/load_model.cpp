/* program used to just load a certain xml model to visually check it */
#include <dart/dart.hpp>
#include <dart/utils/urdf/DartLoader.hpp>
#include <dart/gui/gui.hpp>
#include <dart/gui/SimWindow.hpp>
#include <dart/dynamics/Skeleton.hpp>

using namespace dart::simulation;
using namespace dart::dynamics;

// class MyWindow : public dart::gui::SimWindow {
// public:

//   /// Constructor
//   MyWindow(WorldPtr world) {
//   }
// };

SkeletonPtr create_model(const std::string &path){
    dart::utils::DartLoader loader;
    dart::dynamics::SkeletonPtr model = loader.parseSkeleton(path);
    model->setName("model");

    // Position its base in a reasonable way
    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.translation() = Eigen::Vector3d(-0.65, 0.0, 0.0);
    model->getJoint(0)->setTransformFromParentBodyNode(tf);

    // Get it into a useful configuration
    model->getDof(1)->setPosition(140.0 * M_PI / 180.0);
    model->getDof(2)->setPosition(-140.0 * M_PI / 180.0);

    return model;
}

int main(int argc, char* argv[]) {
    // initialise world
    auto world = dart::simulation::WorldPtr(new dart::simulation::World);

    // load/position the urdf file
    SkeletonPtr model = create_model("/home/medicalrobotics/MASS_EXO/data/exo_model.xml");
    // SkeletonPtr human = create_model("/home/medicalrobotics/MASS_EXO/data/human.xml");
    world->addSkeleton(model);
    // world->addSkeleton(human);
    world->setGravity(Eigen::Vector3d(0,-9.8,0.0));
    world->setTimeStep(1.0/600.0);

    // make window
    dart::gui::SimWindow mywindow;
    mywindow.setWorld(world);

    // begin simulation loop
    glutInit(&argc, argv);
    mywindow.initWindow(3000, 2000, "Model");
    glutMainLoop();
}

