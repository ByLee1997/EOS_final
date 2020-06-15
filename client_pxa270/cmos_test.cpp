// SYSTEM
#include <signal.h>
#include <stdio.h>
#include <strings.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>

// C++ STL
#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>

// PXA related
#include "asm/arch/lib/def.h"
#include "asm/arch/lib/creator_pxa270_lcd.h"
// #include "asm-arm/arch-pxa/lib/creator_pxa270_codec.h"
#include "asm/arch/lib/creator_pxa270_cmos.h"
#include "creator_lib.h"

// Socket
#include "sockop.h"

using namespace std;


#define kFlagSaveImage false // to save image to *.txt file

// CMOS exposure(0~255) & image size setup(CMOS_SIZE_MODE_320_240 or CMOS_SIZE_MODE_160_120)
#define kDesiredImageExposure 160
#define kDesiredImageSizeMode CMOS_SIZE_MODE_320_240
#if kDesiredImageSizeMode == CMOS_SIZE_MODE_320_240
    #define kDesiredImageResolution 320*240
#elif kDesiredImageSizeMode == CMOS_SIZE_MODE_160_120
    #define kDesiredImageResolution 160*120
#else
    #error "Unexpected image size."
#endif

// Global variables
cmos_info_t cmos_info;
unsigned char *image_gray_ptr = NULL;
unsigned char *image_rgb_ptr = NULL;
volatile int fd_cmos = -1;
volatile int fd_socket = -1;
volatile int fd_lcd = -1;

// Socket related
string server_port;
string server_ip;


// Control + C handler
void signal_handler(int signum) {
    cout << "Program shutdown" << endl;

    // Release image memory
    if (image_gray_ptr)
        free(image_gray_ptr);
    if (image_rgb_ptr)
        free(image_rgb_ptr);

    // Release CMOS file descriptor           
    if (fd_cmos < 0){
        ioctl(fd_cmos, CMOS_OFF, &cmos_info);    
        close(fd_cmos); 
    }

    // Release LCD file descriptor           
    if (fd_lcd < 0){ 
        close(fd_cmos); 
    }

    // Release socket file descriptor 
    if (fd_socket != -1){
        close(fd_socket);
        printf("socket close\n");
    }
    exit(0);
}


// Extend gray image to rgb image
int color_interpolate(UC *pOutputBuffer, UC *image_gray_ptr, int nImageWidth, int nImageHeight) {
    int              i, j;
    int              ph ,pw;
    unsigned long    pre_Offset, p_Offset, post_Offset;
    unsigned int     Rval, Gval, Bval ;
    unsigned char    byIRemainder, byJRemainder;
    unsigned char    R, G, B, Counts;
    unsigned char    *cp, *pOutputRGB, *p, *cast;

    ph = nImageHeight; pw = nImageWidth;
    cp = image_gray_ptr; pOutputRGB = pOutputBuffer;
    // Color
    
    for(i=1; i < ph; i++){
        pre_Offset = (i-1)*pw; p_Offset=(i*pw), post_Offset = (i+1)*pw;
        byIRemainder = i & 1;
        for (j=0; j < pw; j++){
            Rval = Gval = Bval = 0;
             byJRemainder = j & 1;

            // At Ri,j --> i%2 == 0 && j%2 == 0
            if (byIRemainder == 0 &&  byJRemainder == 0){
                // R = Ri,j
                // G = (Gri,j-1  + Gri,j+1  + Gbi-1,j  + Gbi+1,j) /4
                // B = (Bi-1,j-1 + Bi-1,j+1 + Bi+1,j-1 + Bi+1,j+1)/4
                R = *cp;

                // Get G value
                Counts = 0;
                if ((j - 1) >= 0){
                    Gval += image_gray_ptr[(p_Offset) +(j-1)];
                    Counts++;
                }
                if ((j+1) < pw){
                    Gval += image_gray_ptr[(p_Offset) +(j+1)];
                    Counts++;
                }
                if ((i - 1) >= 0){
                    Gval += image_gray_ptr[(pre_Offset) +j];
                    Counts++;
                }
                if ((i+1) < ph){
                    Gval += image_gray_ptr[(post_Offset) +j];
                    Counts++;
                }
                G = (unsigned char)(Gval / Counts);

                //Get B
                Counts = 0;
                if ((i-1) >= 0 && (j-1) >= 0){
                    Bval += image_gray_ptr[(pre_Offset) +(j-1)];
                    Counts++;
                }
                if ((i-1) >= 0 && (j+1) < pw){
                    Bval += image_gray_ptr[(pre_Offset) +(j+1)];
                    Counts++;
                }
                if ((i+1) < ph && (j-1) >= 0){
                    Bval += image_gray_ptr[(post_Offset) +(j-1)];
                    Counts++;
                }
                if ((i+1) < ph && (j+1) < pw){
                    Bval += image_gray_ptr[(post_Offset) +(j+1)];
                    Counts++;
                }
                B = (unsigned char)(Bval / Counts);

            }
            //At Gri,j --> i%2 == 0 && j%2 == 1
            else if (byIRemainder == 0 &&  byJRemainder == 1){
                // R = (Ri,j-1 + Ri,j+1) /2
                // G = Gri,j
                // B = (Bi-1,j + Bi+1,j) /2

                // Get R
                Counts = 0;
                if ((j-1) >= 0){
                    Rval +=  image_gray_ptr[(p_Offset)+(j-1)];
                    Counts++;
                }
                if ((j+1) < pw){
                    Rval += image_gray_ptr[(p_Offset)+(j+1)];
                    Counts++;
                }
                R = (unsigned char)(Rval / Counts);

                // Get G
                G = *cp;

                // Get B
                Counts = 0;
                if ((i-1) >= 0){
                    Bval += image_gray_ptr[(pre_Offset)+(j)];
                    Counts++;
                }
                if ((i+1) < ph){
                    Bval += image_gray_ptr[(post_Offset)+(j)];
                    Counts++;
                }
                B = (unsigned char)(Bval / Counts);
            }
            //At Gbi,j --> i%2 == 1 && j%2 == 0
            else if (byIRemainder == 1 &&  byJRemainder == 0){
                // R = (Ri-1,j + Ri+1,j) /2
                // G = Gbi,j
                // B = (Bi,j-1 + Bi,j+1) /2

                // Get R
                Counts  = 0;
                if ((i-1) >= 0){
                    Rval  += image_gray_ptr[(pre_Offset)+(j)];
                    Counts++;
                }
                if ((i+1) < ph){
                    Rval += image_gray_ptr[(post_Offset)+(j)];
                    Counts++;
                }
                R = (unsigned char)(Rval / Counts);

                // Get G
                G = *cp;

                // Get B
                Counts  = 0;
                if ((j-1) >= 0){
                    Bval  += image_gray_ptr[(p_Offset)+(j-1)];
                    Counts++;
                }
                if ((j+1) < pw){
                    Bval += image_gray_ptr[(p_Offset)+(j+1)];
                    Counts++;
                }
                B = (unsigned char)(Bval /Counts);;
            }
            //At Bi,j --> i%2 == 1 && j%2 == 1
            else{
                // R = (Ri-1,j-1 + Ri-1,j+1 + Ri+1,j-1 + Ri+1,j+1) /4
                // G = (Gbi,j-1  + Gbi,j+1  + Gri-1,j  + Gri+1,j)  /4
                // B = Bi,j;

                // Get R
                Counts  = 0;
                if ((i-1) >= 0 && (j-1) >= 0){
                    Rval += image_gray_ptr[(pre_Offset) +(j-1)];
                    Counts++;
                }
                if ((i-1) >= 0 && (j +1) < pw){
                    Rval += image_gray_ptr[(pre_Offset) +(j+1)];
                    Counts++;
                }
                if ((i+1) < ph && (j-1) >= 0){
                    Rval += image_gray_ptr[(post_Offset) +(j-1)];
                    Counts++;
                }
                if ((i+1) < ph && (j+1) < pw){
                    Rval += image_gray_ptr[(post_Offset) +(j+1)];
                    Counts++;
                }
                R = (unsigned char)(Rval /Counts);

                //Get G
                Counts  = 0;
                if ((j-1) >= 0){
                    Gval += image_gray_ptr[(p_Offset) +(j-1)];
                    Counts++;
                }
                if ((j+1) < pw){
                    Gval += image_gray_ptr[(p_Offset) +(j+1)];
                    Counts++;
                }
                if ((i-1) >= 0){
                    Gval += image_gray_ptr[(pre_Offset) +j];
                    Counts++;
                }
                if ((i+1) < ph){
                    Gval += image_gray_ptr[(post_Offset) +j];
                    Counts++;
                }
                G = (unsigned char)(Gval / Counts);;

                // Get B
                B = *cp;
            }
            cp++;

            *pOutputRGB++ = R;
            *pOutputRGB++ = G;
            *pOutputRGB++ = B;                
        }//pw
    }//ph     

    //compensate line 0
    p =  pOutputBuffer + (nImageWidth *3);  // line 1
    cast = pOutputBuffer;                   // line 0
    for (j=0; j < nImageWidth; j++) {
        *cast++=(*p++); *cast++=(*p++); *cast++=(*p++);
    }
    
    //compensate line (nImageHeight-1)
    p =  pOutputBuffer + ((nImageHeight - 2)* nImageWidth *3);   // line Height-2       
    cast = pOutputBuffer + ((nImageHeight - 1)* nImageWidth *3); // line Height-1   
    for (j=0; j< nImageWidth; j++){ 
        *cast++=(*p++); *cast++=(*p++); *cast++=(*p++);
    }

    return 0;
}


void cmos_init(volatile int* fd_cmos_ptr, cmos_info_t* cmos_info_ptr) {
    // Open CMOS file descriptor
    *fd_cmos_ptr = open("/dev/cmos", O_RDWR);
    if (*fd_cmos_ptr < 0) {
        fprintf(stderr, "Open ccm error\nAny key to exit:");   
        exit(-1);    
    }

    // Create image buffer
    image_gray_ptr = (unsigned char*)malloc(kDesiredImageResolution);
    image_rgb_ptr = (unsigned char*)malloc(kDesiredImageResolution * 3);
    if (image_gray_ptr == NULL || image_rgb_ptr == NULL){
        fprintf(stderr, "mem alloc error\nAny key to exit:");
        close(*fd_cmos_ptr);
        exit(-1);                 
    }

    // Set camera parameters
    cmos_info_ptr->ImageSizeMode = kDesiredImageSizeMode;
    cmos_info_ptr->HighRef = kDesiredImageExposure;
    if (ioctl(*fd_cmos_ptr, CMOS_PARAMETER, cmos_info_ptr) < 0) {
        fprintf(stderr, "Set CMOS camera parameters failed!\n");
        close(*fd_cmos_ptr);
        exit(-1);
    }
}

unsigned int cnt_save_image = 0;
void *capture_image(void *userdata) {
    // Remenber userdata=[&fd_socket, &fd_cmos, &cmos_info]
    int* fd_socket_ptr = (int*)((int*)userdata)[0];
    int* fd_cmos_ptr = (int*)((int*)userdata)[1];
    cmos_info_t* cmos_info_ptr = (cmos_info_t*)((int*)userdata)[2];

    int image_width = cmos_info_ptr->ImageWidth;
    int image_height = cmos_info_ptr->ImageHeight;      
    int image_resolution = image_width * image_height;

    // Loop for reading image from CMOS
    while(1) {
        int total_read_size = 0; 
        int count = image_resolution;
        do {
            if (count + total_read_size > image_resolution)
                count = image_resolution - total_read_size ;
            int num_read = read(*fd_cmos_ptr, image_gray_ptr+total_read_size, count);
            if (num_read > 0 ) 
                total_read_size += num_read;
            else if(num_read == 0)  /* EOF */
                break;
            else {
                fprintf(stderr, "Reading CMOS camera data failed!\n");
                close(*fd_cmos_ptr);
                exit(1);
            }
        } while(total_read_size < image_resolution);

        /* write the image data to a file */
        if(kFlagSaveImage == true) {
            FILE *image_fd;
            char file_idx[10];
            sprintf(file_idx, "%d", ++cnt_save_image);
            string output_image_name = "myimage" +  string(file_idx) + ".txt";
            image_fd = fopen(output_image_name.c_str(), "w");
            for(int i = 0; i < cmos_info.ImageHeight; i++) {
                for(int j =0; j < cmos_info.ImageWidth; j++) {
                int k = i* cmos_info.ImageWidth + j;
                fprintf(image_fd,"%d ", image_gray_ptr[k]);
                }
                fprintf(image_fd,"\n");
            }
            fclose(image_fd);
        }

        if(write(*fd_socket_ptr, image_gray_ptr, image_resolution) < 0){
            fprintf(stderr, "Image transimission error\n");
            exit(-1);
        }
    }
    return 0;
}


int main(int argc, char *argv[]) {
    // Register signal handler
    signal(SIGINT, signal_handler);

    if(argc != 3) {
        fprintf(stderr, "Usage: ./cmos_test <server_ip> <server_port>\n");
        exit(-1);
    }
    server_ip = argv[1];
    server_port = argv[2];
    fd_socket = connectsock(server_ip.c_str(), server_port.c_str(), "tcp");

    // Open LCD
    fd_lcd = open("/dev/lcd", O_RDWR);
    if (fd_lcd < 0){
        fprintf(stderr, "open /dev/lcd error\n");
        exit(-1);
    }
    KEYPAD_clear_buffer(fd_lcd);
    LCD_fClearScreen(fd_lcd);

    // CMOS initialization
    cmos_init(&fd_cmos, &cmos_info);

    // Turn on CMOS camera
    if (ioctl(fd_cmos, CMOS_ON, &cmos_info) < 0){     
        fprintf(stderr, "Turn on CMOS camera failed!\n");
        close(fd_cmos);
        exit(-1);
    }

    // Check whether the total data size is correct or not
    int image_width = cmos_info.ImageWidth;
    int image_height = cmos_info.ImageHeight;      
    int image_resolution = image_width * image_height;
    printf("Captured image size: %dx%d\n", image_width, image_height);
    if(image_resolution != kDesiredImageResolution) {  
        fprintf(stderr, "Image size returned is different from assigned!\n");
        close(fd_cmos);
        exit(1);
    }

    // Wait for the image to be ready
    do {
        // Get camera status
        if (ioctl(fd_cmos, CMOS_GET_STATUS, &cmos_info) < 0) {
            fprintf(stderr, "Accessing CMOS camera control failed!\n");
            close(fd_cmos);
            exit(1);
        }
        // If image not ready yet, give more time
        if (cmos_info.Status == CMOS_IMG_EMPTY) 
            usleep(100000);
    } while(cmos_info.Status != CMOS_IMG_READY);


    // Create another thread to capture CMOS image
    // Note that: pointer_array=[&fd_socket, &fd_cmos, &cmos_info]
    pthread_t tid;
    int *pointer_array[3];
    pointer_array[0] = (int*)&fd_socket; 
    pointer_array[1] = (int*)&fd_cmos; 
    pointer_array[2] = (int*)&cmos_info;
    if(pthread_create(&tid, NULL, capture_image, (void*)pointer_array) < 0){
        fprintf(stderr, "pthread created error");
        exit(-1);
    }
    sleep(60);

    // Turn off CMOS camera 
    if (ioctl (fd_cmos, CMOS_OFF, &cmos_info) < 0) {  /* turn off camera */
        fprintf(stderr,"Turn off CMOS camera failed!\n");
        close(fd_cmos);
        exit(1);
    }

    // Release resource
    if (image_gray_ptr)
        free(image_gray_ptr);
    if (image_rgb_ptr)
        free(image_rgb_ptr);            
    if (fd_cmos < 0){
        ioctl(fd_cmos, CMOS_OFF, &cmos_info);    
        close(fd_cmos); 
    }
    if (fd_socket != -1){
        close(fd_socket);
        printf("socket close\n");
    }
    return 0;
}
