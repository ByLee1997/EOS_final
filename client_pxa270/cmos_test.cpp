// SYSTEM
#include <signal.h>
#include <stdio.h>
#include <strings.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/fcntl.h>
#include <errno.h>
#include <unistd.h>

// C++ STL
#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>

// PXA related
#include "asm-arm/arch-pxa/lib/def.h"
#include "asm-arm/arch-pxa/lib/creator_pxa270_lcd.h"
// #include "asm-arm/arch-pxa/lib/creator_pxa270_codec.h"
#include "asm-arm/arch-pxa/lib/creator_pxa270_cmos.h"
#include "creator_lib.h"

// Socket
#include "sockop.h"

using namespace std;

// Global variables
unsigned char   *pImageBuffer=NULL, *pRGBBuffer=NULL;
int fd_cmos = -1, fd_socket = -1, fd_lcd = -1;

void signal_handler(int signum) {
    cout << "Program shutdown" << endl;
    cmos_info_t CMOSInfo;  
    if (pImageBuffer)
        free(pImageBuffer);
    if (pRGBBuffer)
        free(pRGBBuffer);            
    if (fd_cmos < 0){
        ioctl(fd_cmos, CMOS_OFF, &CMOSInfo);    
        close(fd_cmos); 
    }
    if (fd_socket != -1){
        close(fd_socket);
        printf("socket close\n");
    }
    exit(0);
}


int main(int argc, char *argv[]) {
    cmos_info_t CMOSInfo;   
    int         ret =OK, WaitImage;   
    int         nRead, nWrite, terminate, CMOSRunning, ImgW, ImgH;      
    unsigned long   dwImageSize, dwTotalReadSize, count ;
    unsigned short  key;
    char *pMsg = "\nImage to PC...";

    // Register signal handler
    signal(SIGINT, signal_handler);

    // Open LCD
    fd_lcd = open("/dev/lcd", O_RDWR);
    if (fd_lcd < 0){
        printf("open /dev/lcd error\n");
        return (-1);
    }
    KEYPAD_clear_buffer(fd_lcd);
    LCD_fClearScreen(fd_lcd);

    // Open CMOS
    fd_cmos = open("/dev/cmos", O_RDWR);
    if (fd_cmos < 0) {
        LCD_ErrorMessage(fd_lcd, "Open ccm error\nAny to exit:");   
        return -1;
    }

    // Create image buffer
    pImageBuffer = (unsigned char*)malloc(320*240);
    pRGBBuffer = (unsigned char*)malloc(320*240*3);
    if (pImageBuffer == NULL || pRGBBuffer == NULL){
        LCD_ErrorMessage(fd_lcd, "mem alloc error\nAny to exit:");      
        return -1;                  
    }

    CMOSInfo.command = PC_CMOS_ON;
    ret = ioctl(fd_cmos, CMOS_ON, &CMOSInfo);               
    if (ret < 0){
        LCD_ErrorMessage(fd_lcd, "CCM ON error\n\n Any to exit:");      
        return -1;               
    }

    ImgW = CMOSInfo.ImageWidth ; ImgH = CMOSInfo.ImageHeight ;      
    dwImageSize = ImgW * ImgH ;
    ioctl(fd_cmos, CMOS_PARAMETER, &CMOSInfo);

    WaitImage = 1;
    while(WaitImage){   
        if (ioctl(fd_cmos, CMOS_GET_STATUS, &CMOSInfo) >= 0){
            if (CMOSInfo.Status == CMOS_IMG_READY){
                dwTotalReadSize = 0; count = dwImageSize;       
                do {            
                   if (count + dwTotalReadSize > dwImageSize)
                       count = dwImageSize - dwTotalReadSize ;
    
                   nRead = read(fd_cmos, pImageBuffer+dwTotalReadSize, count);  
                   if (nRead > 0 ){
                       dwTotalReadSize += nRead;
                   }
                   else if (nRead == 0){
                       break;   
                   }
                   else{
                       //break;
                   }
                } while (dwTotalReadSize < dwImageSize); 
                  
                nWrite = write(fd_socket, pImageBuffer, dwImageSize);                     
                WaitImage = 0;                      
            }
        }
        if (KEYPAD_get_key(fd_lcd, &key) == OK){              
            return -1;  
        }               
        else if (CMOSInfo.Status == CMOS_IMG_EMPTY){
            //Delay(200);           
        }
    }
    ioctl(fd_cmos, CMOS_OFF, &CMOSInfo);

    terminate = 0;      
    CMOSRunning = 0;
    printf("%s", pMsg);
    LCD_printf("%s", pMsg); 
    while (terminate){  
        /* µ¥«ÝPC ¶Ç°ecommand */    
        nRead = recv(fd_socket, &CMOSInfo, sizeof(cmos_info_t), 0);
        if (nRead == 0){
            LCD_ErrorMessage(fd_lcd, "socket break\n\n Any to exit:");              
            terminate = 0;
        }    
        
        if (CMOSInfo.command == PC_CMOS_ON){    
            ret = ioctl(fd_cmos, CMOS_ON, &CMOSInfo);               
            if (ret < 0){
                LCD_ErrorMessage(fd_lcd, "CCM ON error\n\n Any to exit:");      
                return -1;               
            }
            ImgW = CMOSInfo.ImageWidth ; ImgH = CMOSInfo.ImageHeight ;      
            dwImageSize = ImgW * ImgH ;
            continue ;
        }
        else if (CMOSInfo.command == PC_CMOS_OFF){      
            ioctl(fd_cmos, CMOS_OFF, &CMOSInfo);    
            return -1;  
        }       
        ioctl(fd_cmos, CMOS_PARAMETER, &CMOSInfo);      
    }

    if (pImageBuffer)
        free(pImageBuffer);
    if (pRGBBuffer)
        free(pRGBBuffer);            
    if (fd_cmos < 0){
        ioctl(fd_cmos, CMOS_OFF, &CMOSInfo);    
        close(fd_cmos); 
    }
    if (fd_socket != -1){
        close(fd_socket);
        printf("socket close\n");
    }
    return 0;
}
