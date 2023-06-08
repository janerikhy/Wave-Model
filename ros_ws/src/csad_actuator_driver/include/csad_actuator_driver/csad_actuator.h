#include <string>
#include "dynamixel_sdk/dynamixel_sdk.h"

#ifndef CSAD_ACTUATOR_DRIVER_H
#define CSAD_ACTUATOR_DRIVER_H

//  PWM CONSTS
#define MODE1               0x00        //Mode  register  1
#define MODE2               0x01	    //Mode  register  2
#define SUBADR1             0x02        //I2C-bus subaddress 1
#define SUBADR2             0x03        //I2C-bus subaddress 2
#define SUBADR3             0x04        //I2C-bus subaddress 3
#define ALLCALLADR          0x05        //LED All Call I2C-bus address
#define LED0                0x6         //LED0 start register
#define LED0_ON_L           0x6		    //LED0 output and brightness control byte 0
#define LED0_ON_H           0x7		    //LED0 output and brightness control byte 1
#define LED0_OFF_L          0x8		    //LED0 output and brightness control byte 2
#define LED0_OFF_H          0x9		    //LED0 output and brightness control byte 3
#define LED_MULTIPLYER      4	        // For the other 15 channels
#define ALLLED_ON_L         0xFA        //load all the LEDn_ON registers, byte 0 (turn 0-7 channels on)
#define ALLLED_ON_H         0xFB	    //load all the LEDn_ON registers, byte 1 (turn 8-15 channels on)
#define ALLLED_OFF_L        0xFC	    //load all the LEDn_OFF registers, byte 0 (turn 0-7 channels off)
#define ALLLED_OFF_H        0xFD	    //load all the LEDn_OFF registers, byte 1 (turn 8-15 channels off)
#define PRE_SCALE           0xFE		//prescaler for output frequency
#define CLOCK_FREQ          25000000.0  //25MHz default osc clock
#define PWM_FREQ            100         //PWM freq
#define PCA_I2C_ADDRESS     0x40        // I2C address of the PCA9685 module
#define I2C_DEVICE          "/dev/i2c-"
//! Main class that exports features for PCA9685 chip

//  SERVO CONSTS
// Control table address
#define SERVO_ADDR_CW_ANGLE_LIMIT   6           //addresses that change servo settings
#define SERVO_ADDR_CCW_ANGLE_LIMIT  8
#define SERVO_ADDR_TORQUE_ENABLE    24
#define SERVO_ADDR_GOAL_POSITION    30
#define SERVO_ADDR_PRESENT_POSITION 36        
// Protocol version
#define SERVO_PROTOCOL_VERSION      1.0       // Default Protocol version of DYNAMIXEL MX series.
// Default setting
#define SERVOBAUDRATE               57600           // Default Baudrate of DYNAMIXEL X series
#define USB_DEVICE                 "/dev/ttyUSB"   // [Linux] To find assigned port, use "$ ls /dev/ttyUSB*" command
#define SERVO_RESOLUTION            4096            // number of divisions per 180deg / pi rad
#define NUMBER_OF_SERVOS            6               // total number of servos on the dynamix servo network

//  GENERIC CONSTS
#define PI                          3.141593        // PI



class CSAD_Actuator
{
public:
    int numberOfServos;

    CSAD_Actuator();
    double getServoPresentPosition(uint8_t servoNumber);
    void getAllServoPresentPositions(double positions[]);
    void setServoPosition(double position, uint8_t servoNumber);
    void setAllServoPositions(double positions[NUMBER_OF_SERVOS]);
    void closeI2CPort();
    void resetPCAModule();
    void setMotorPower(uint8_t motor, double power);
    void setAllMotorPower(double power[NUMBER_OF_SERVOS]);
    void setPWMFreq(int freq);
    void setPWM(uint8_t motor, int on_value, double power);
    int getPWM(uint8_t led);
};

#endif  // CSAD_ACTUATOR_DRIVER_H