# Request

#
# This message is used to asynchronously report the execution result of a power mode change
# requested via the PowerModeChangeSrv.srv service. It must be used in conjunction with PowerModeChangeSrv.srv.
#
# Example workflow:
#   1. A node calls the PowerModeChangeSrv.srv service to request a mode change (e.g., CMD_SLEEP).
#   2. The service returns immediately with acceptance status (RET_SUCCESS or RET_ERROR).
#   3. The power management module performs the operation asynchronously.
#   4. Upon completion or failure, this message is published on: PowerModeChangeResult.msg
#
# This enables non-blocking service calls while still providing result feedback.


string requester_node_name      # Name of the requesting node
uint8  cmd                      # The requested power mode command

# Execution result status code
uint8 result_code

# Optional description (e.g., error message)
string message


# Constants for cmd
uint8 CMD_SHUTDOWN   = 0        # Shut down the robot
uint8 CMD_SLEEP      = 1        # Enter sleep mode
uint8 CMD_WORK       = 2        # Enter work mode
uint8 CMD_EMERGENCY  = 3        # Emergency stop (cut off the power supply to the joint motor)
uint8 CMD_RESTART    = 4        # Restart the system
uint8 CMD_CHARGE_REQ = 5        # Trigger charging back to the pile via APP or remote control
uint8 CMD_EMERGENCY_CANCLE = 6  # Cancel emergency stop (restore the power supply to the joint motor)

# Result code constants
uint8 RESULT_SUCCESS  = 0       # Mode switch completed successfully
uint8 RESULT_ERROR    = 1       # Operation failed