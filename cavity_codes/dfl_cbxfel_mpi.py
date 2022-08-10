import numpy as np
import time, os, sys
from rfp2 import *
from Bragg_mirror import *
import time
import pickle
from mpi4py import MPI
import psutil
import pickle
from pathlib import Path

def propagate_slice_kspace(field, z, xlamds, kx, ky):
    H = np.exp(-1j*xlamds*z*(kx**2 + ky**2)/(4*np.pi))
    return field*H

def Bragg_mirror_reflect(ncar, dgrid, xlamds, nslice, dt, npadx=0, 
                         verboseQ = True, showPlotQ = False, xlim = None, ylim = None):
    t0 = time.time()
    
    h_Plank = 4.135667696e-15;      # Plank constant [eV-sec]
    c_speed  = 299792458;           # speed of light[m/sec]
    
    # get photon energy coordinate
    hw0_eV = h_Plank * c_speed / xlamds
    Dhw_eV = h_Plank / dt 
    dhw_eV = Dhw_eV / (nslice - 1.)
    eph = hw0_eV + Dhw_eV / 2. * np.linspace(-1.,1.,nslice)
    
    # get transverse angle coordinate
    theta_0 = 45.0*np.pi/180.
    dx = 2. * dgrid / ncar
    Dkx = 2. * np.pi / dx
    Dtheta = Dkx * xlamds / 2. / np.pi
    theta = theta_0 + Dtheta / 2. * np.linspace(-1.,1.,ncar+2*int(npadx))
    

   
    R0H = Bragg_mirror_reflection(eph, theta).T
    
    R00 = Bragg_mirror_transmission(eph, theta).T
        
    if verboseQ: print('took',time.time()-t0,'seconds to calculate Bragg filter')


    if showPlotQ:  # plot reflectivity and transmission
        
        # axes
        pi = np.pi; ncontours = 100
        thetaurad = 1e6*(theta-pi/4)
        Eph,Thetaurad = np.meshgrid(eph,thetaurad);
        
        
        # contour plots vs hw and kx
        absR2 = np.abs(R0H.T)**2
        absRT = np.abs(R00.T)**2
        print('absR2.shape =',absR2.shape)
        print('np.sum(np.isnan(absR2.reshape(-1))) =',np.sum(np.isnan(absR2.reshape(-1))))
        print('np.sum(absR2.reshape(-1)>0) =',np.sum(absR2.reshape(-1)>0))
        
        
        plt.figure(1)
        plt.contourf(Eph,Thetaurad,absR2,ncontours, label='reflectivity')
        #plt.contour(np.fft.fftshift(Eph),np.fft.fftshift(Thetaurad),absR2,3)
        plt.ylabel('Angle - 45 deg (urad)')
        plt.xlabel('Photon energy (eV)')
        
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        
        plt.colorbar()
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        plt.figure(2)
        plt.contourf(Eph,Thetaurad,absRT,ncontours, label='reflectivity')
        #plt.contour(np.fft.fftshift(Eph),np.fft.fftshift(Thetaurad),absR2,3)
        plt.ylabel('Angle - 45 deg (urad)')
        plt.xlabel('Photon energy (eV)')

        
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        
        plt.colorbar()
        plt.legend()
        plt.tight_layout()
        plt.show()
      
        
        # slice plot vs hw along kx=0
        x, _  = Eph.shape
        plt.figure(3)
        plt.plot(Eph[x//2,:],np.abs(R0H.T[x//2, :])**2,label='reflectivity')
        plt.plot(Eph[x//2, :],np.abs(R00.T[x//2, :])**2,label='transmission')
        plt.title(['Angle = 45 deg'])
        plt.xlabel('Photon energy (eV)')
        if xlim: 
            plt.xlim(xlim)
        plt.ylim([0,1])
        plt.legend()
        plt.tight_layout()
        plt.show()
        #fwhm1 = half_max_x(Eph[cut], np.abs(R.T[cut])**2)
        #print("FWHM is" + str(fwhm1) + 'eV')
        
    
    return R0H, R00


def propagate_slice(fld_slice, npadx,     # fld slice in spectral space, (Ek, x, y)
                             R00_slice, R0H_slice,     # Bragg reflection information
                             l_cavity, l_undulator, w_cavity,  # cavity parameter
                             lambd_slice, kx_mesh, ky_mesh, xmesh, ymesh, #fld slice information
                             roundtripQ,               # recirculation parameter
                             verboseQ): 
    
    # propagate one slice around the cavity
    # take a slice in real space, unpadded, return a slice in real space, unpadded
    
    
     # focal length of the lens
    flens1 = (l_cavity + w_cavity)/2
    flens2 = (l_cavity + w_cavity)/2
    
    # propagation length in cavity
    z_und_start = (l_cavity - l_undulator)/2
    z_und_end = z_und_start + l_undulator
    

        
    # pad in x
    if npadx > 0:
        fld_slice = pad_dfl_slice_x(fld_slice, [int(npadx),int(npadx)])
   
    # fft to kx, ky space
    t0 = time.time()
    fld_slice = np.fft.fftshift(fft2(fld_slice), axes=(0,1))
    if verboseQ: print('took',time.time()-t0,'seconds for fft over x, y')
    
    # if a roundtrip, propagate from UNDSTART to UNDEND
    if roundtripQ:
        fld_slice = propagate_slice_kspace(field = fld_slice, z = l_undulator, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
    # drift from UNDEND to M1
    Ldrift = l_cavity - z_und_end
        
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)

    
        
    # reflect from M1
    fld_slice = np.einsum('i,ij->ij',R0H_slice,fld_slice)
    # trasmission through M1
    #fld_slice = np.einsum('i,ij->ij',R00_slice,fld_slice)
        
        
    # drift to the lens
    Ldrift = w_cavity/2
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
        
    # lens
    f = flens1
    #ifft to the real space
    fld_slice = ifft2(np.fft.ifftshift(fld_slice))
    #apply intracavity focusing CRL
    fld_slice *= np.exp(-1j*np.pi/(f*lambd_slice)*(xmesh**2 + ymesh**2))
    #fft to kx, ky space, check it!!!!
    fld_slice = np.fft.fftshift(fft2(fld_slice))
   
        
        
    # drift to M2
    Ldrift = w_cavity/2
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
        
    # reflect from M2
    fld_slice = np.einsum('i,ij->ij',np.flip(R0H_slice),fld_slice)
        
    # drift to M3
    Ldrift = l_cavity
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
    # reflect from M3
    fld_slice = np.einsum('i,ij->ij',np.flip(R0H_slice),fld_slice)
        
    # drift to lens
    Ldrift = w_cavity/2
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
    # lens
    f = flens2
    #ifft to the real space
    fld_slice = ifft2(np.fft.ifftshift(fld_slice))
    #apply intracavity focusing CRL
    fld_slice *= np.exp(-1j*np.pi/(f*lambd_slice)*(xmesh**2 + ymesh**2))
    #fft to kx, ky space, check it!!!!
    fld_slice = np.fft.fftshift(fft2(fld_slice))
  
    # drift to M4
    Ldrift = w_cavity/2
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
    # reflect from M4
    fld_slice = np.einsum('i,ij->ij',np.flip(R0H_slice),fld_slice)
        
    # drift to undulator start
    Ldrift = z_und_start
    fld_slice = propagate_slice_kspace(field = fld_slice, z = Ldrift, xlamds = lambd_slice, kx = kx_mesh, ky = ky_mesh)
        
    # recirculation finished, ifft to real space
    fld_slice = ifft2(np.fft.ifftshift(fld_slice))
   
        
    # unpad in x
    if npadx > 0:
        fld_slice = unpad_dfl_slice_x(fld_slice,  [int(npadx),int(npadx)])

    
    return fld_slice

def recirculate_to_undulator_mpi(zsep, ncar, dgrid, nslice, xlamds=1.261043e-10,           # dfl params
                             npadt = 0, Dpadt = 0, npadx = 0,isradi = 1,       # padding params
                             l_undulator = 32*3.9, l_cavity = 149, w_cavity = 1,  # cavity params
                             showPlotQ = False, savePlotQ = False, verboseQ = 1, # plot params
                             nRoundtrips = 0,                     # recirculation params
                             readfilename = None, writefilename = None):        # read and write
    
    t00 = time.time()
    
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    
    
    h_Plank = 4.135667696e-15      # Plank constant [eV-sec]
    c_speed  = 299792458           # speed of light[m/sec]
    
    dt = xlamds*zsep * max(1,isradi) /c_speed
    
    nslice_padded = nslice + 2*int(npadt)
    nx = ny = ncar
    nx_padded = ncar + 2*int(npadx)
    
    
    #------------------------------
    # get the size of each sub-task
    #------------------------------
    ave, res = divmod(nslice_padded, nprocs)
    count = [ave + 1 if p < res else ave for p in range(nprocs)]
    count = np.array(count)
    count_sum = [sum(count[:p]) for p in range(nprocs)]
    displ = [sum(count[:p])*nx*ny for p in range(nprocs)]
        
    if rank == 0:
        print("Input data split into vectors of sizes %s" %count)
        print("Input data split with displacements of %s" %displ)
        print("Input data split with displacements in row of %s" %count_sum)
    
    #-------------------------------
    # get coordinates after padding
    #-------------------------------

    # get photon energy coordinate
    hw0_eV = h_Plank * c_speed / xlamds
    Dhw_eV = h_Plank / dt 
    eph = hw0_eV + Dhw_eV / 2. * np.linspace(-1.,1., nslice_padded)
    lambd = h_Plank*c_speed/eph

    # get kx,ky coordinates
    dx = 2. * dgrid / ncar
    Dkx = 2. * np.pi / dx
    kx = Dkx/ 2. * np.linspace(-1.,1.,nx_padded)
    ky = Dkx/ 2. * np.linspace(-1.,1.,ny)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    kx_mesh = kx_mesh.T
    ky_mesh = ky_mesh.T

    # get x, y coordinates
    xs = (np.arange(nx_padded) - np.floor(nx_padded/2))*dx
    ys = (np.arange(ny) - np.floor(ny/2))*dx
    xmesh, ymesh = np.meshgrid(xs, ys)
    xmesh = xmesh.T
    ymesh = ymesh.T

    #----------------------------
    # get Bragg mirror response
    #----------------------------    

    R0H, R00 = Bragg_mirror_reflect(ncar = ncar, dgrid = dgrid, xlamds = xlamds, nslice = nslice_padded, dt = dt, npadx=npadx, 
                             verboseQ = True, showPlotQ = showPlotQ, xlim = [9831,9833], ylim = [-10, 10])    
    
    
    #-------------------------------------------------------------------------------------------
    # read or make field on root node
    #------------------------------------------------------------------------------------------- 
    if readfilename == None:
        saveFilenamePrefix = 'test'
        workdir = '.'
    else:
        saveFilenamePrefix = readfilename
        temp = readfilename.split('/')[:-1]
        workdir = "/".join(temp)
    
    if rank == 0:                    
        if readfilename == None:
            # make a new field
            t0 = time.time()
            fld = make_gaus_beam(ncar= ncar, dgrid=dgrid, w0=40e-6, dt=dt, nslice=nslice, trms=10.e-15)
            print('took',time.time()-t0,'seconds total to make field with dimensions',fld.shape)
            fld = fld[::isradi,:,:]
            print("fld shape after downsample ", fld.shape)

        else:
            # read dfl file on disk
            print('Reading in',readfilename)
            t0 = time.time()
            fld = read_dfl(readfilename, ncar=ncar,conjugate_field_for_genesis=False, swapxyQ=False) # read the field from disk
            print('took',time.time()-t0,'seconds total to read in and format the field with dimensions',fld.shape)
            fld = fld[:nslice,:,:]
            print('fld shape after truncation ', fld.shape)
            fld = fld[::isradi,:,:]
            print("fld shape after downsample ", fld.shape)

        if showPlotQ:
            # plot the imported field
            plot_fld_marginalize_t(fld, dgrid, dt=dt, saveFilename=saveFilenamePrefix+'_init_xy.png',showPlotQ=showPlotQ, savePlotQ = savePlotQ) 
            plot_fld_slice(fld, dgrid, dt=dt, slice=-2, saveFilename=saveFilenamePrefix+'_init_tx.png',showPlotQ=showPlotQ, savePlotQ = savePlotQ)
            plot_fld_slice(fld, dgrid, dt=dt, slice=-1, saveFilename=saveFilenamePrefix+'_init_ty.png',showPlotQ=showPlotQ, savePlotQ = savePlotQ)
            plot_fld_power(fld, dt=dt, saveFilename=saveFilenamePrefix+'_init_t.png',showPlotQ=showPlotQ, savePlotQ = savePlotQ)


        energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm = fld_info(fld, dgrid = dgrid, dt=dt)

        init_field_info = [energy_uJ, maxpower, trms, tfwhm, xrms, xfwhm, yrms, yfwhm]
    
        #--------------------------------------------------
        # fft in time domain to get spectral representaion
        #--------------------------------------------------
        # pad field in time
        if int(npadt) > 0:
            fld = pad_dfl_t(fld, [int(npadt),int(npadt)])
            if verboseQ: print('Padded field in time by',int(npadt),'slices (',dt*int(npadt)*1e15,'fs) at head and tail')
        #nslice_padded, nx, ny = fld.shape
        if verboseQ:
            print("after padding, fld shape " + str(fld.shape))
        # plot the field after padding
        if showPlotQ:
            plot_fld_marginalize_t(fld, dgrid) 
            plot_fld_slice(fld, dgrid, dt=dt, slice=-2) 
            plot_fld_slice(fld, dgrid, dt=dt, slice=-1)

        # fft
        t0 = time.time()
        fld = np.fft.fftshift(fft(fld, axis=0), axes=0)
        if verboseQ: print('took',time.time()-t0,'seconds for fft over t')
        
    #-------------------------------------------------------------------------------------------
    # create variables on other nodes
    #------------------------------------------------------------------------------------------- 
    else:
        fld = None
        # initialize count on worker processes
        #count = np.zeros(nprocs, dtype=np.int)
        #displ = None
    
    # broadcast count
    #comm.Bcast(count, root=0)
    #comm.Barrier()
    
    #---------------------------------------------------------------------------------------------
    # scatter fld along first axis to all workers
    #---------------------------------------------------------------------------------------------
    
    # initialize recvbuf on all processes
    recvbuf_real  = np.zeros((count[rank],nx,ny))
    recvbuf_imag  = np.zeros((count[rank],nx,ny))
    
    comm.Scatterv([np.ascontiguousarray(np.real(fld)), count*nx*ny, displ, MPI.DOUBLE], recvbuf_real, root=0)
    comm.Barrier()
    
    comm.Scatterv([np.ascontiguousarray(np.imag(fld)), count*nx*ny, displ, MPI.DOUBLE], recvbuf_imag, root=0)
    comm.Barrier()
    
    #print('After Scatterv, process {} has real data shape :'.format(rank), recvbuf_real.shape)
    #print('After Scatterv, process {} has imag data shape :'.format(rank), recvbuf_imag.shape)
    
    comm.Barrier()
    
    # put together the real and imag data
    fld_block = recvbuf_real + 1j* recvbuf_imag 
    
    
    recvbuf_real = None
    recvbuf_imag = None
    
    
    #---------------------------------------------------------------------------------------------------
    # propagate slice by slice 
    # TODO: 
    #    1. angular error, wavefront distort
    #    2. delete plot function
    #    3. add transmission
    #    4. delete writing to disk for the last roundtrip
    #    5. rotate the power peak to center 
    #---------------------------------------------------------------------------------------------------
    
    # first round from Undstart to Undend
    t0 = time.time()
    for k in range(fld_block.shape[0]):   
        #if k%50 == 0:    
            #print("worker " + str(rank) + " finished "+str(np.round(k/fld_block.shape[0],2)*100) + " % of the job")
        
        # take the frequency slice
        fld_slice = np.squeeze(fld_block[k, :, :])
        
        # take the reflectivity and transmission slice
        ind0 = count_sum[rank]
        R00_slice = np.squeeze(R00[ind0+k, :])
        R0H_slice = np.squeeze(R0H[ind0+k, :])
        lambd_slice = lambd[ind0+k]
        
       
        
        # propagate the slice from und end to und start
        fld_slice = propagate_slice(fld_slice = fld_slice, npadx = npadx,     
                             R00_slice = R00_slice, R0H_slice = R0H_slice,     
                             l_cavity = l_cavity, l_undulator = l_undulator, w_cavity = w_cavity,  
                             lambd_slice = lambd_slice, kx_mesh = kx_mesh, ky_mesh = ky_mesh, xmesh = xmesh, ymesh = ymesh, 
                             roundtripQ = False, verboseQ = False)
       
        # record the current slice
        fld_block[k,:, :] = fld_slice
    
    pickle.dump(fld_block, open(saveFilenamePrefix +"_block"+str(rank)+"_round0.p", "wb" ) ) 
    
    #For additional roundtrips
    for l in range(nRoundtrips):
        for k in range(fld_block.shape[0]):
            # take the frequency slice
            fld_slice = np.squeeze(fld_block[k, :, :])
        
            # take the reflectivity and transmission slice
            ind0 = count_sum[rank]
            R00_slice = np.squeeze(R00[ind0+k, :])
            R0H_slice = np.squeeze(R0H[ind0+k, :])
            lambd_slice = lambd[ind0+k]
        
       
        
           # propagate the slice from und start to und start
            fld_slice = propagate_slice(fld_slice = fld_slice, npadx = npadx,     
                             R00_slice = R00_slice, R0H_slice = R0H_slice,     
                             l_cavity = l_cavity, l_undulator = l_undulator, w_cavity = w_cavity,  
                             lambd_slice = lambd_slice, kx_mesh = kx_mesh, ky_mesh = ky_mesh, xmesh = xmesh, ymesh = ymesh, 
                             roundtripQ = True, verboseQ = False)
       
            # record the current slice
            fld_block[k,:, :] = fld_slice
        
        pickle.dump(fld_block, open(saveFilenamePrefix + "_block"+str(rank)+"_round"+str(l+1)+".p", "wb" ) ) 
            
        
       
    if rank < nprocs - 1:
        print('slice #', count_sum[rank], ' to slice #', count_sum[rank+1], ' finished by worker ' ,rank, ', took',time.time()-t0, 'seconds')
    else:
        print('slice #', count_sum[rank], ' to slice #', np.sum(count),  ' finished by worker ', rank, ', took',time.time()-t0, 'seconds')
    
    
    comm.Barrier()
    
    #-------------------------------------------------------------------------------------------------
    # gather fld_blocks back to the root node
    #-------------------------------------------------------------------------------------------------
    
    sendbuf2_real = np.ascontiguousarray(np.real(fld_block))
    sendbuf2_imag = np.ascontiguousarray(np.imag(fld_block))
    
    if rank ==0:
        recvbuf2_real = np.zeros((nslice_padded, nx, ny))
        recvbuf2_imag = np.zeros((nslice_padded, nx, ny))

    else:
        recvbuf2_real = None
        recvbuf2_imag = None
    
    comm.Gatherv(sendbuf2_real, [recvbuf2_real, count*nx*ny, displ, MPI.DOUBLE], root=0)
    comm.Barrier()
    comm.Gatherv(sendbuf2_imag, [recvbuf2_imag, count*nx*ny, displ, MPI.DOUBLE], root=0)
    comm.Barrier()
    

    
    #-------------------------------------------------------------------------------------------------
    # inverse fft in time on the root node
    #-------------------------------------------------------------------------------------------------
    
    if rank == 0:
        fld = recvbuf2_real + 1j* recvbuf2_imag

        #----------------------
        # ifft to time domain
        #----------------------
        t0 = time.time()
        fld = ifft(np.fft.ifftshift(fld,axes = 0), axis=0)
        if verboseQ: print('took',time.time()-t0,'seconds for ifft over t')

        #----------------
        # Dpadt in time
        #----------------
        if int(Dpadt) > 0:

            fld = unpad_dfl_t(fld, [int(Dpadt), int(Dpadt)])
            print("shape of fld after unpadding is ", fld.shape)

            if verboseQ: print('Removed padding of ',dt*int(npadt)*1e15,'fs in time from head and tail of field')



        #-----------------------------------------   
        # plot the final result and write results
        #-----------------------------------------
        if showPlotQ:
            plot_fld_marginalize_t(fld, dgrid)
            plot_fld_slice(fld, dgrid, dt=dt, slice=-2)
            plot_fld_slice(fld, dgrid, dt=dt, slice=-1)

        # write field to disk
        if writefilename != None:
            print('Writing to',writefilename)
                #writefilename = readfilename + 'r'
            write_dfl(fld, writefilename,conjugate_field_for_genesis = False,swapxyQ=False)

        print('It takes ' + str(time.time() - t00) + ' seconds to finish the recirculation.') 
    
    #-----------------------------------------------------------------------------------------
    # merge files from each roundtrip
    #-----------------------------------------------------------------------------------------
    if nRoundtrips > 0:
        t0 = time.time()
        ave2, res2 = divmod(nRoundtrips + 1, nprocs)
        count2 = [ave2 + 1 if p < res2 else ave2 for p in range(nprocs)]
        count2 = np.array(count2)
        count_sum2 = [sum(count2[:p]) for p in range(nprocs)]
        if count2[rank] > 0:
            if rank == 0:
                Round_range = range(0, count_sum2[0])
            else:
                Round_range = range(count_sum2[rank-1], count_sum2[rank])
            for Round in Round_range:
                field_t = []
                for block in range(nprocs):
                    loadname = saveFilenamePrefix + "_block"+str(block)+"_round"+str(Round)+".p"
                    field_t.append(pickle.load( open(loadname , "rb" )))
                field_t = np.concatenate(field_t, axis = 0)
    
                writefilename = saveFilenamePrefix+"_field_round" + str(Round)
                write_dfl(field_t, writefilename,conjugate_field_for_genesis = False,swapxyQ=False)
    
    comm.Barrier()
    if rank == 0:
        for filename in Path(workdir).glob("*.p"):
            filename.unlink()
        print("It takes " + str(time.time()-t0) + "seconds to finish merging files")

            
    return fld 
        
        
if __name__ == '__main__':
    ncar = 256
    dgrid = 400e-6
    w0 =40e-6
    xlamds = 1.261043e-10
    zsep = 200
    c_speed  = 299792458
    nslice = 1024
    npadt = 2048
    npadx = 512
    showPlotQ = False
    savePlotQ = False
    verbosity = True
    isradi = 1
    fld = recirculate_to_undulator_mpi(zsep = zsep, ncar = ncar, dgrid = dgrid, nslice = nslice, xlamds=1.261043e-10,           # dfl params
                                 npadt = npadt, Dpadt = 0, npadx = npadx,isradi = isradi,       # padding params
                                 l_undulator = 32*3.9, l_cavity = 149, w_cavity = 1,  # cavity params
                                 showPlotQ = showPlotQ, savePlotQ = savePlotQ, verboseQ = 1, # plot params
                                 nRoundtrips = 10,               # recirculation params
                                 readfilename = None, writefilename = './test1.dfl')
    
    
