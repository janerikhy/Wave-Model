
#include <ros/ros.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <stdio.h>
#include <fcntl.h>
#include <syslog.h>
#include <inttypes.h>
#include <errno.h>
#include <math.h>
#include <string>

#include "dynamixel_sdk/dynamixel_sdk.h"
#include "csad_actuator_driver/csad_actuator.h"

using namespace dynamixel;

double servoSignalToRad(int servoSignal, double offset)
{
  return (-servoSignal * 2.0 * PI / (double)SERVO_RESOLUTION) - offset;
}

int radToServoSignal(double rad, double offset)
{
  return (-((rad - offset) * (double)SERVO_RESOLUTION / (2.0 * PI) + 0.5));
}

PortHandler *portHandler;
PacketHandler *packetHandler;
int fd_;
double servoOffsets[] = {-0.52f, 1.475f, -1.568f, 1.1f, -0.02f, 1.156f}; // The offset in the first element here corresponds to the offset of the servo with the id matching the first element of servoIds
int servoIds[] = {1, 2, 3, 4, 5, 6};

/**
 * @brief Constructor
 */
CSAD_Actuator::CSAD_Actuator()
{
  uint8_t dxl_error = 0;
  int dxl_comm_result = COMM_TX_FAIL;

  // loops through different ubs files to look for the U2D2 module.
  for (int i = 0; i < 6; i++)
  {
    std::string path = USB_DEVICE + std::to_string(i);
    fd_ = open(path.c_str(), O_RDWR);
    if (fd_ < 0)
    {
      ROS_INFO("falied to open :%s", path.c_str());
    }
    else
    {
      ROS_INFO("opened :%s", path.c_str());
      portHandler = PortHandler::getPortHandler(path.c_str());                 // opens the file name for communication through usb
      packetHandler = PacketHandler::getPacketHandler(SERVO_PROTOCOL_VERSION); // sets the comunication protocoll version of the Servos
      break;
    }
  }

  // sets communication baud rate between the servos and the U2D2 dongle
  if (!portHandler->setBaudRate(SERVOBAUDRATE))
  {
    ROS_ERROR("Failed to set the baudrate!");
  }
  else
  {
    ROS_INFO("Baudrate set");
  }

  for (int i = 0; i < NUMBER_OF_SERVOS; i++)
  {
    // enables torque for each servo
    dxl_comm_result = packetHandler->write1ByteTxRx(
        portHandler, servoIds[i], SERVO_ADDR_TORQUE_ENABLE, 1, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
      ROS_ERROR("Failed to enable torque for Dynamixel ID %d", servoIds[i]);
    }
    // sets the servos in multirotor mode
    dxl_comm_result = packetHandler->write1ByteTxRx(
        portHandler, servoIds[i], SERVO_ADDR_CW_ANGLE_LIMIT, 4095, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
      ROS_ERROR("Failed to set clockwise limit for Dynamixel ID %d", servoIds[i]);
    }
    else
    {
      ROS_INFO("successfully set clockwise limit for Dynamixel ID %d", servoIds[i]);
    }
    dxl_comm_result = packetHandler->write1ByteTxRx(
        portHandler, servoIds[i], SERVO_ADDR_CCW_ANGLE_LIMIT, 4095, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
      ROS_ERROR("Failed to set counterclockwise limit for Dynamixel ID %d", servoIds[i]);
    }
    else
    {
      ROS_INFO("successfully set counterclockwise limit for Dynamixel ID %d", servoIds[i]);
    }
  }
  // opens the PCA9685 module
  for (int i = 0; i < 6; i++)
  {
    std::string path = I2C_DEVICE + std::to_string(i);
    fd_ = open(path.c_str(), O_RDWR);
    if (fd_ < 0)
    {
      ROS_INFO("falied to open :%s", path.c_str());
    }
    else
    {
      ROS_INFO("I2C port opened on: %s", path.c_str());
      break;
    }
  }
  ioctl(fd_, I2C_SLAVE, PCA_I2C_ADDRESS);

  setPWMFreq(PWM_FREQ);
}

/**
 * @brief get current Position of specific servo
 * @param id uint8_t id of the servo with the position you want to read
 * @retval position -4096 is negative 180deg + 4095 is positive 180deg
 */
double CSAD_Actuator::getServoPresentPosition(uint8_t servoNum)
{
  uint8_t dxl_error = 0;
  int dxl_comm_result = COMM_TX_FAIL;

  // Position Value of X series is 4 byte data. For AX & MX(1.0) use 2 byte data(int16_t) for the Position Value.
  int16_t position = 0;

  // Read Present Position (length : 4 bytes) and Convert uint32 -> int32
  // When reading 2 byte data from AX / MX(1.0), use read2ByteTxRx() instead.
  dxl_comm_result = packetHandler->read2ByteTxRx(
      portHandler, servoIds[servoNum - 1], SERVO_ADDR_PRESENT_POSITION, (uint16_t *)&position, &dxl_error);
  if (dxl_comm_result == COMM_SUCCESS)
  {
    ROS_INFO("getPosition : [ID:%d] -> [POSITION:%d]", servoIds[servoNum - 1], servoSignalToRad(position, servoOffsets[servoNum - 1]));
    return (servoSignalToRad(position, servoOffsets[servoNum - 1]));
  }
  else
  {
    ROS_INFO("Failed to get position! Result: %d", dxl_comm_result);
  }
}

void CSAD_Actuator::getAllServoPresentPositions(double positions[])
{
  uint8_t dxl_error = 0;
  int dxl_comm_result = COMM_TX_FAIL;
  int16_t position = 0;

  // Read Present Position (length : 2 bytes) and Convert uint32 -> int32
  for (int i = 0; i < NUMBER_OF_SERVOS; i++)
  {
    dxl_comm_result = packetHandler->read2ByteTxRx(
        portHandler, servoIds[i], SERVO_ADDR_PRESENT_POSITION, (uint16_t *)&position, &dxl_error);
    if (dxl_comm_result == COMM_SUCCESS)
    {
      ROS_INFO("getPosition : [ID:%d] -> [POSITION:%f]", servoIds[i], (float)servoSignalToRad(position, servoOffsets[i]));
      positions[i] = servoSignalToRad(position, servoOffsets[i]);
    }
    else
    {
      ROS_INFO("Failed to get position! Result: %d", dxl_comm_result);
    }
  }
}

/**
 * @brief set position of single servo
 * @param position position you want the servo set to
 * @param servoNumber servonumber of the servo you want to set the position of
 */
void CSAD_Actuator::setServoPosition(double position, uint8_t servoNumber)
{
  uint8_t dxl_error = 0;
  int dxl_comm_result = COMM_TX_FAIL;

  // Write Goal Position (length : 2 bytes)
  dxl_comm_result = packetHandler->write2ByteTxRx(
      portHandler, servoIds[servoNumber - 1], SERVO_ADDR_GOAL_POSITION, radToServoSignal(position, servoOffsets[servoNumber - 1]), &dxl_error);
  if (dxl_comm_result == COMM_SUCCESS)
  {
    ROS_INFO("setPosition : [ID:%d] [POSITION:%d]", servoNumber, radToServoSignal(position, servoOffsets[servoNumber - 1]));
  }
  else
  {
    ROS_ERROR("Failed to set position! Result: %d,", dxl_comm_result);
  }
}

/**
 * @brief set position of all servos
 * @param positions positions you want the servo set to
 */
void CSAD_Actuator::setAllServoPositions(double positions[])
{
  uint8_t dxl_error = 0;
  int dxl_comm_result = COMM_TX_FAIL;

  // Write Goal Position (length : 2 bytes)
  for (int i = 0; i < NUMBER_OF_SERVOS; i++)
  {
    dxl_comm_result = packetHandler->write2ByteTxRx(
        portHandler, servoIds[i], SERVO_ADDR_GOAL_POSITION, radToServoSignal(positions[i], servoOffsets[i]), &dxl_error);
    if (dxl_comm_result == COMM_SUCCESS)
    {
      ROS_INFO("setPosition : [ID:%d] [POSITION:%d] ,   %f", servoIds[i], radToServoSignal(positions[i], servoOffsets[i]), positions[i]);
    }
    else
    {
      ROS_ERROR("Failed to set position! Result: %d", dxl_comm_result);
    }
  }
}

/**
 * @brief Closes port.. duh
 */
void CSAD_Actuator::closeI2CPort()
{
  close(fd_);
}

/**
 * @brief resets the PCA module
 */
void CSAD_Actuator::resetPCAModule()
{

  unsigned char buff[2];
  buff[0] = MODE1;
  buff[1] = 0x00;
  write(fd_, buff, 2); // Normal mode
  buff[0] = MODE2;
  buff[1] = 0x04;
  write(fd_, buff, 2); // totem pole
}

/**
 * @brief sets the frequency of the PWM signal.
 * @param freq desired PWM frequency in Hz.
 */
void CSAD_Actuator::setPWMFreq(int freq)
{
  uint8_t prescale_val = (CLOCK_FREQ / 4096 / freq) - 1;
  unsigned char buff[2];
  buff[0] = MODE1;
  buff[1] = 0x10;
  write(fd_, buff, 2); // sleep
  buff[0] = PRE_SCALE;
  buff[1] = prescale_val;
  write(fd_, buff, 2); // multiplyer for PWM frequency
  buff[0] = MODE1;
  buff[1] = 0x80;
  write(fd_, buff, 2); // restart
  buff[0] = MODE2;
  buff[1] = 0x04;
  write(fd_, buff, 2); // totem pole (default)
}

/**
 @brief sets the PWM signal on time.
 @param motor motor number to contrll power to
 @param power power to set the motor to. -1.0 being 100% reverse 1.0 being 100% forward.
 */
void CSAD_Actuator::setMotorPower(uint8_t motor, double power)
{
  setPWM(motor, 0, (power));
}

/**
 @brief sets the PWM signal on time.
 @param motor motor number to contrll power to
 @param power power to set the motor to. -1.0 being 100% reverse 1.0 being 100% forward.
 */
void CSAD_Actuator::setAllMotorPower(double power[])
{
  for (int i = 0; i < 6; i++)
  {
    setPWM(i + 1, 0, (power[i]));
  }
}

void CSAD_Actuator::setPWM(uint8_t motor, int on_value, double power)
{
  // int off_value = (int)(power * 5*51.2f) + 307.2f; //round to nearest int and map from +- 1 to 6.25 and 8.75%
  int off_value = (int)((power * 205.0f) + 614.0f);
  ROS_INFO("pwm signal = %d", off_value);
  unsigned char buff[2];
  buff[0] = LED0_ON_L + LED_MULTIPLYER * (motor - 1);
  buff[1] = on_value & 0xFF;
  write(fd_, buff, 2);
  buff[0] = LED0_ON_H + LED_MULTIPLYER * (motor - 1);
  buff[1] = on_value >> 8;
  write(fd_, buff, 2);
  buff[0] = LED0_OFF_L + LED_MULTIPLYER * (motor - 1);
  buff[1] = off_value & 0xFF;
  write(fd_, buff, 2);
  buff[0] = LED0_OFF_H + LED_MULTIPLYER * (motor - 1);
  buff[1] = off_value >> 8;
  write(fd_, buff, 2);
}

/**
 @brief Get current PWM value on specified pin.
 @param led specify pin 1-16
 */
int CSAD_Actuator::getPWM(uint8_t led)
{
  int ledval = 0;
  unsigned char buff[1];
  buff[0] = LED0_OFF_H + LED_MULTIPLYER * (led - 1);
  read(fd_, buff, 1);
  ledval = buff[0];
  ledval = ledval & 0xf;
  ledval <<= 8;
  buff[0] = LED0_OFF_L + LED_MULTIPLYER * (led - 1);
  ledval += buff[0];
  return ledval;
}
