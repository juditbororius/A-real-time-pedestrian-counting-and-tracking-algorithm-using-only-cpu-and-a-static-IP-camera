import Variables as var
from functions import *

starttime = time.time()

while (var.cap.isOpened):

    ret, frame = var.cap.read()

    if ret == True:

        df = pd.DataFrame()
        bbinfo = pd.DataFrame(columns=['Frame', 'ID', 'maxID', 'date', 'time', 'topleft', 'width', 'height', 'center', 'velocity'])

        var.q.append(frame)
        if len(var.q) == int(var.ufps):


            medianFrame, contours = creating_countours(var.q)

            realFrame = cv2.resize(var.q[var.memory_frames-1], var.dim)

            #imgplot = plt.imshow(realFrame)
            #plt.show()

            var.ident, curr, centers, bbinfo, count_people, fg_bounding = contour_calculations(contours,
                                                                                               var.ident,
                                                                                               medianFrame,
                                                                                               realFrame,
                                                                                               bbinfo,
                                                                                               var.Icount,
                                                                                               var.memory)

            shop_in_space, shop_out_space = creation_in_out_spaces(var.veritas)
            pts = np.array(shop_in_space, np.int32)
            pts1 = np.array(shop_out_space, np.int32)
            pts = pts.reshape((-1,1,2))
            pts1 = pts1.reshape((-1, 1, 2))
            f = cv2.resize(frame, var.dim)
            #cv2.polylines(realFrame, [pts], True, (255, 0, 0), thickness = 2)
            #cv2.polylines(realFrame, [pts1], True, (0, 255, 0), thickness=2)
            #cv2.imwrite('rectangles.jpg', f)
            #break

            df = df.append({'Frame': var.Icount, 'Count': count_people}, ignore_index=True)

            info = [
                ("In", var.total_in),
                ("Out", var.total_out),
                ("Izquierda", var.total_izquierda),
                ("Derecha", var.total_derecha),
                ("Total counted", var.ident-1),
                ("Now counted", count_people),
            ]

            info_derecha = [
                ('In', var.in_derecha),
                ('Out', var.out_derecha),
            ]

            info_izquierda = [
                ('In', var.in_izquierda),
                ('Out', var.out_izquierda),
            ]

            finaltime = time.time()
            dif = finaltime - starttime
            #cv2.putText(medianFrame, 'Video time: {:.2f} and frame: {}'.format(dif, var.Icount),
            #(0, 15), 0, 0.5, (0, 0, 0), 2)


            # loop over the info tuples and draw them on our frame
            cv2.rectangle(realFrame, (0, var.dim[1]-(((len(info)+1)*20))), (len('Total counted')*15, var.dim[1]), (255,255,255), thickness=-1)
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(realFrame, text, (10, var.dim[1] - ((i * 20) + 20)),
                            16, 0.5, (0, 0, 0), 1)

            cv2.rectangle(realFrame, (var.W-len('counted')*15, 200), (var.W, 200+(((len(info_derecha)+1)*20))), (255,255,255), thickness=-1)
            for (i, (k, v)) in enumerate(info_derecha):
                text = "{}: {}".format(k, v)
                cv2.putText(realFrame, text, (var.W-len('counted')*15+10, (200+(((len(info_derecha)+1)*20))) - ((i * 20) + 20)),
                            16, 0.5, (0, 0, 0), 1)

            cv2.rectangle(realFrame, (0, 200),
                          (len('counted') * 15, 200 + (((len(info_izquierda) + 1) * 20))), (255, 255, 255), thickness=-1)
            for (i, (k, v)) in enumerate(info_izquierda):
                text = "{}: {}".format(k, v)
                cv2.putText(realFrame, text, (10, (200 + (((len(info_izquierda) + 1) * 20))) - ((i * 20) + 20)),
                            16, 0.5, (0, 0, 0), 1)


            temp = pd.merge(df, bbinfo, on='Frame')
            var.new = var.new.append(temp)
            var.new.to_excel('DataOutputs/new.xlsx', index = False)

            var.out.write(realFrame)
            cv2.imshow('frame', realFrame)

            cv2.imwrite('{}/bounding{}.jpg'.format(var.directory, var.Icount), fg_bounding)
            cv2.imwrite('{}/frame{}.jpg'.format(var.directory, var.Icount), realFrame)

            var.frame_array.append(realFrame)
            var.Icount += 1
            var.q.clear()
            if var.Icount > 1:
                if len(var.memory) == var.memory_frames:
                    # if var.Icount != (((Nframes - Nframes%var.ufps)/var.ufps)-1):
                    var.memory.append(curr)
                    var.memory.popleft()
                else:
                    var.memory.append(curr)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break


    else:
        break

# print(var.frame_array)
var.cap.release()
var.out.release()
cv2.destroyAllWindows()

#clip = mv.ImageSe.quenceClip(var.frame_array, fps=var.ufps)
#clip.write_videofile('shortest.mp4')
