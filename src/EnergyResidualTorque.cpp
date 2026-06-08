#include "EnergyResidualTorque.h"

#include <mc_control/GlobalPluginMacros.h>

namespace mc_plugin
{

EnergyResidualTorque::~EnergyResidualTorque() = default;

void EnergyResidualTorque::init(mc_control::MCGlobalController & controller, const mc_rtc::Configuration & config)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);

  auto & robot = ctl.robot(ctl.robots()[0].name());
  auto & realRobot = ctl.realRobot(ctl.robots()[0].name());
  auto & rjo = robot.refJointOrder();

  // Make sure to have obstacle detection
  if(!ctl.controller().datastore().has("Obstacle detected"))
  {
    ctl.controller().datastore().make<bool>("Obstacle detected", false);
  }

  ctl.controller().datastore().make<bool>("Energy Residual Torque Obstacle detected", false);

  dt_ = ctl.timestep();
  counter_ = 0.0;
  jointNumber = ctl.robot(ctl.robots()[0].name()).refJointOrder().size();

  auto plugin_config = config("energy_residual");
  ko = plugin_config("ko", 1.0);
  threshold_filtering_ = plugin_config("threshold_filtering", 0.05);
  threshold_offset_ = plugin_config("threshold_offset", 2.0);
  lpf_threshold_.setValues(threshold_offset_, threshold_filtering_, jointNumber);

  Eigen::VectorXd qdot(jointNumber);
  for(size_t i = 0; i < jointNumber; i++)
  {
    qdot[i] = robot.alpha()[robot.jointIndexByName(rjo[i])][0];
  }
  residual = 0.0;
  residual_high_ = 0.0;
  residual_low_ = 0.0;
  integralTerm = 0.0;
  coriolis = new rbd::Coriolis(robot.mb());
  forwardDynamics = rbd::ForwardDynamics(robot.mb());

  forwardDynamics.computeH(robot.mb(), robot.mbc());
  auto inertiaMatrix = forwardDynamics.H() - forwardDynamics.HIr();
  tzero = 0.5 * qdot.transpose() * inertiaMatrix * qdot;

  addGui(ctl);
  addLog(ctl);

  mc_rtc::log::info("EnergyResidualTorque::init called with configuration:\n{}", config.dump(true, true));
}

void EnergyResidualTorque::reset(mc_control::MCGlobalController & controller)
{
  // mc_rtc::log::info("EnergyResidualTorque::reset called");
}

void EnergyResidualTorque::before(mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);
  counter_ += dt_;

  if(activate_plot_ && !plot_added_)
  {
    addPlot(controller);
    plot_added_ = true;
  }

  if(ctl.controller().datastore().has("Zurlo Collision Detection"))
  {
    collision_stop_activated_zurlo_ = ctl.controller().datastore().get<bool>("Zurlo Collision Detection");
  }

  energy_residual_computation(controller);
  residual_high_ = lpf_threshold_.adaptiveThreshold(residual, true);
  residual_low_ = lpf_threshold_.adaptiveThreshold(residual, false);
  obstacle_detected_ = false;
  if(residual > residual_high_ || residual < residual_low_)
  {
    obstacle_detected_ = true;
    if(activate_verbose) mc_rtc::log::info("[Energy Residual Torque] Obstacle detected");
    if(collision_stop_activated_)
    {
      ctl.controller().datastore().get<bool>("Obstacle detected") = obstacle_detected_;
    }
  }
  ctl.controller().datastore().get<bool>("Energy Residual Torque Obstacle detected") = obstacle_detected_;
}

void EnergyResidualTorque::after(mc_control::MCGlobalController & controller)
{
  // mc_rtc::log::info("EnergyResidualTorque::after");
}

mc_control::GlobalPlugin::GlobalPluginConfiguration EnergyResidualTorque::configuration()
{
  mc_control::GlobalPlugin::GlobalPluginConfiguration out;
  out.should_run_before = true;
  out.should_run_after = false;
  out.should_always_run = false;
  return out;
}

void EnergyResidualTorque::energy_residual_computation(mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);

  auto & robot = ctl.robot(ctl.robots()[0].name());
  auto & realRobot = ctl.realRobot(ctl.robots()[0].name());
  auto & rjo = robot.refJointOrder();

  Eigen::VectorXd tau_m = Eigen::VectorXd::Map(realRobot.jointTorques().data(), realRobot.jointTorques().size());

  Eigen::VectorXd qdot(jointNumber);
  rbd::paramToVector(realRobot.alpha(), qdot);
 
  forwardDynamics.computeC(realRobot.mb(), realRobot.mbc());
  forwardDynamics.computeH(realRobot.mb(), realRobot.mbc());
  auto coriolisMatrix = coriolis->coriolis(realRobot.mb(), realRobot.mbc());
  auto coriolisGravityTerm = forwardDynamics.C();
  auto negative_gravity = coriolisMatrix*qdot - coriolisGravityTerm;
  auto inertiaMatrix = forwardDynamics.H() - forwardDynamics.HIr();
  double t_kinetic = 0.5 * qdot.transpose() * inertiaMatrix * qdot;

  integralTerm += (qdot.transpose()*(tau_m + negative_gravity) + residual) * ctl.timestep();
  residual = ko * (t_kinetic - integralTerm - tzero);

  if(!ctl.controller().datastore().has("energy_residual"))
  {
    ctl.controller().datastore().make<double>("energy_residual", residual);
  }
  else
  {
    ctl.controller().datastore().assign("energy_residual", residual);
  }
}

void EnergyResidualTorque::addGui(mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);

  ctl.controller().gui()->addElement({"Plugins", "EnergyResidualTorque"},
  mc_rtc::gui::NumberInput(
      "Gain", [this]() { return this->ko; },
      [this](double gain)
      {
      this->integralTerm = 0.0;
      this->residual = 0.0;
      this->ko = gain;
      }),
      mc_rtc::gui::Button("Add plot", [this]() { return activate_plot_ = true; }),
    // Add checkbox to activate the collision stop
    mc_rtc::gui::Checkbox("Collision stop", collision_stop_activated_),
    mc_rtc::gui::Checkbox("Verbose", activate_verbose), 
    // Add Threshold offset input
    mc_rtc::gui::NumberInput("Threshold offset", [this](){return this->threshold_offset_;},
        [this](double offset)
      { 
        threshold_offset_ = offset;
        lpf_threshold_.setOffset(threshold_offset_); 
      }),
    // Add Threshold filtering input
    mc_rtc::gui::NumberInput("Threshold filtering", [this](){return this->threshold_filtering_;},
        [this](double filtering)
      { 
        threshold_filtering_ = filtering;
        lpf_threshold_.setFiltering(threshold_filtering_); 
      })       
    );

}

void EnergyResidualTorque::addLog(mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);

  ctl.controller().logger().addLogEntry("EnergyResidualTorque_residual", [this]() -> const double & { return residual; });
  ctl.controller().logger().addLogEntry("EnergyResidualTorque_threshold_high", [this]() -> const double & { return residual_high_; });
  ctl.controller().logger().addLogEntry("EnergyResidualTorque_threshold_low", [this]() -> const double & { return residual_low_; });
  ctl.controller().logger().addLogEntry("EnergyResidualTorque_threshold_offset", [this]() -> const double & { return threshold_offset_; });
  ctl.controller().logger().addLogEntry("EnergyResidualTorque_threshold_filtering", [this]() -> const double & { return threshold_filtering_; });
  ctl.controller().logger().addLogEntry("EnergyResidualTorque_ko", [this]() -> const double & { return ko; });
  ctl.controller().logger().addLogEntry("EnergyResidualTorque_obstacleDetected", [this]() -> const bool & { return obstacle_detected_; });
}

void EnergyResidualTorque::addPlot(mc_control::MCGlobalController & controller)
{
  auto & ctl = static_cast<mc_control::MCGlobalController &>(controller);
  auto & gui = *ctl.controller().gui();

  gui.addPlot(
    "EnergyResidualTorque",
    mc_rtc::gui::plot::X("t", [this]() { return counter_; }),
    mc_rtc::gui::plot::Y("Residual", [this]() { return residual; }, mc_rtc::gui::Color::Red),
    mc_rtc::gui::plot::Y("Threshold high", [this]() { return residual_high_; }, mc_rtc::gui::Color::Green),
    mc_rtc::gui::plot::Y("Threshold low", [this]() { return residual_low_; }, mc_rtc::gui::Color::Blue)
  );
}

} // namespace mc_plugin

EXPORT_MC_RTC_PLUGIN("EnergyResidualTorque", mc_plugin::EnergyResidualTorque)