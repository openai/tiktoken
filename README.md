#sudo su 
@ -1,9 +1,5 @@
import os
import sys
import posixpath
import time
import datetime
@ -11,26 +7,33 @@ import glob
import shutil
import fnmatch
import zipfile
if os.name == 'nt':
    import msvcrt
elif os.name == 'posix':
    import fcntl
try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    import json as simplejson
except ImportError:
    import simplejson
import email
import email.Utils
import email.Generator
import email.Message
import email.header
import email.encoders
import smtplib
import ftplib
import socket
import ssl
from django.utils.translation import ugettext as _
#Bots modules
import botslib
import botsglobal
from botsconfig import *

@botslib.log_session
def run(idchannel,command,idroute,rootidta=None):
@ -42,7 +45,7 @@ def run(idchannel,command,idroute,rootidta=None):
                                WHERE idchannel=%(idchannel)s''',
                                {'idchannel':idchannel}):
        channeldict = dict(row)   #convert to real dictionary ()
        botsglobal.logger.debug(u'Start communication channel "%(idchannel)s" type %(type)s %(inorout)s.',channeldict)
        #for acceptance testing bots has an option to turn of external communication in channels
        if botsglobal.ini.getboolean('acceptance','runacceptancetest',False):
            #override values in channels for acceptance testing.
@ -56,7 +59,7 @@ def run(idchannel,command,idroute,rootidta=None):
                    channeldict['type'] = 'mimefile'
                else:   #channeldict['type'] in ['ftp','ftps','ftpis','sftp','xmlrpc','ftp','ftp','communicationscript','db',]
                    channeldict['type'] = 'file'
            botsglobal.logger.debug(u'Channel "%(idchannel)s" adapted for acceptance test: type "%(type)s", testpath "%(testpath)s".',channeldict)
                
        #update communication/run process with idchannel
        ta_run = botslib.OldTransaction(botslib._Transaction.processlist[-1])
@ -79,10 +82,10 @@ def run(idchannel,command,idroute,rootidta=None):

        comclass = classtocall(channeldict,idroute,userscript,scriptname,command,rootidta) #call the class for this type of channel
        comclass.run()
        botsglobal.logger.debug(u'Finished communication channel "%(idchannel)s" type %(type)s %(inorout)s.',channeldict)
        break   #there can only be one channel; this break takes care that if found, the 'else'-clause is skipped
    else:
        raise botslib.CommunicationError(_(u'Channel "%(idchannel)s" is unknown.'),{'idchannel':idchannel})


class _comsession(object):
@ -109,14 +112,14 @@ class _comsession(object):
        else:   #incommunication
            if self.command == 'new': #only in-communicate for new run
                #handle maxsecondsperchannel: use global value from bots.ini unless specified in channel. (In database this is field 'rsrv2'.)
                self.maxsecondsperchannel = botsglobal.ini.getint('settings','maxsecondsperchannel',sys.maxint) if self.channeldict['rsrv2'] <= 0 else self.channeldict['rsrv2']
                try:
                    self.connect()
                except:     #in-connection failed. note that no files are received yet. useful if scheduled quite often, and you do nto want error-report eg when server is down. 
                    #max_nr_retry : get this from channel. should be integer, but only textfields where left. so might be ''/None->use 0
                    max_nr_retry = int(self.channeldict['rsrv1']) if self.channeldict['rsrv1'] else 0
                    if max_nr_retry:
                        domain = u'bots_communication_failure_' + self.channeldict['idchannel']
                        nr_retry = botslib.unique(domain)  #update nr_retry in database
                        if nr_retry >= max_nr_retry:
                            botslib.unique(domain,updatewith=0)    #reset nr_retry to zero
@ -128,13 +131,14 @@ class _comsession(object):
                    #max_nr_retry : get this from channel. should be integer, but only textfields where left. so might be ''/None->use 0
                    max_nr_retry = int(self.channeldict['rsrv1']) if self.channeldict['rsrv1'] else 0
                    if max_nr_retry:
                        domain = u'bots_communication_failure_' + self.channeldict['idchannel']
                        botslib.unique(domain,updatewith=0)    #set nr_retry to zero 
                self.incommunicate()
                self.disconnect()
            self.postcommunicate()
            self.archive()


    def archive(self):
        ''' after the communication channel has ran, archive received of send files.
            archivepath is the root directory for the archive (for this channel).
@ -189,26 +193,27 @@ class _comsession(object):
            if archiveexternalname:
                if self.channeldict['inorout'] == 'in':
                    # we have internal filename, get external
                    absfilename = botslib.abspathdata(row['filename'])
                    taparent = botslib.OldTransaction(idta=row['idta'])
                    ta_list = botslib.trace_origin(ta=taparent,where={'status':EXTERNIN})
                    if ta_list:
                        archivename = os.path.basename(ta_list[-1].filename)
                    else:
                        archivename = row['filename']
                else:
                    # we have external filename, get internal
                    archivename = os.path.basename(row['filename'])
                    taparent = botslib.OldTransaction(idta=row['idta'])
                    ta_list = botslib.trace_origin(ta=taparent,where={'status':FILEOUT})
                    absfilename = botslib.abspathdata(ta_list[0].filename)
            else:
                # use internal name in archive
                absfilename = botslib.abspathdata(row['filename'])
                archivename = os.path.basename(row['filename'])

            if self.userscript and hasattr(self.userscript,'archivename'):
                archivename = botslib.runscript(self.userscript,self.scriptname,'archivename',channeldict=self.channeldict,idta=row['idta'],filename=absfilename)
            #~print 'archive',os.path.basename(absfilename),'as',archivename

            if archivezip:
                archivezipfilehandler.write(absfilename,archivename)
@ -221,6 +226,7 @@ class _comsession(object):
        if archivezip and checkedifarchivepathisthere:
            archivezipfilehandler.close()


    def postcommunicate(self):
        pass

@ -233,7 +239,7 @@ class _comsession(object):
            from status FILEOUT to FILEOUT
        '''
        #select files with right statust, status and channel.
        for row in botslib.query('''SELECT idta,filename,frompartner,topartner,charset,contenttype,editype
                                    FROM ta
                                    WHERE idta>%(rootidta)s
                                    AND status=%(status)s
@ -243,58 +249,53 @@ class _comsession(object):
                                    {'idchannel':self.channeldict['idchannel'],'status':FILEOUT,
                                    'statust':OK,'idroute':self.idroute,'rootidta':self.rootidta}):
            try:
                ta_from = botslib.OldTransaction(row['idta'])
                ta_to = ta_from.copyta(status=FILEOUT)
                ta_to.synall()  #needed for user exits: get all parameters of ta_to from database;
                confirmtype = u''
                confirmasked = False
                charset = row['charset']

                if row['editype'] == 'email-confirmation': #outgoing MDN: message is already assembled
                    outfilename = row['filename']
                else:   #assemble message: headers and payload. Bots uses simple MIME-envelope; by default payload is an attachment
                    message = email.Message.Message()
                    #set 'from' header (sender)
                    frommail,ccfrom_not_used_variable = self.idpartner2mailaddress(row['frompartner'])    #lookup email address for partnerID
                    message.add_header('From', frommail)

                    #set 'to' header (receiver)
                    if self.userscript and hasattr(self.userscript,'getmailaddressforreceiver'):    #user exit to determine to-address/receiver
                        tomail,ccto = botslib.runscript(self.userscript,self.scriptname,'getmailaddressforreceiver',channeldict=self.channeldict,ta=ta_to)
                    else:
                        tomail,ccto = self.idpartner2mailaddress(row['topartner'])          #lookup email address for partnerID
                    message.add_header('To',tomail)
                    if ccto:
                        message.add_header('CC',ccto)

                    if botsglobal.ini.getboolean('acceptance','runacceptancetest',False):
                        reference = '123message-ID email should be unique123'
                        email_datetime = email.Utils.formatdate(timeval=time.mktime(time.strptime("2013-01-23 01:23:45", "%Y-%m-%d %H:%M:%S")),localtime=True)
                    else:
                        reference = email.Utils.make_msgid(unicode(ta_to.idta))    #use transaction idta in message id.
                        email_datetime = email.Utils.formatdate(localtime=True)
                    message.add_header('Message-ID',reference)
                    message.add_header("Date",email_datetime)
                    ta_to.update(frommail=frommail,tomail=tomail,cc=ccto,reference=reference)   #update now (in order to use correct & updated ta_to in userscript)

                    #set Disposition-Notification-To: ask/ask not a a MDN?
                    if botslib.checkconfirmrules('ask-email-MDN',idroute=self.idroute,idchannel=self.channeldict['idchannel'],
                                                                frompartner=row['frompartner'],topartner=row['topartner']):
                        message.add_header("Disposition-Notification-To",frommail)
                        confirmtype = u'ask-email-MDN'
                        confirmasked = True

                    #set subject
                    if botsglobal.ini.getboolean('acceptance','runacceptancetest',False):
                        subject = u'12345678'
                    else:
                        subject = unicode(row['idta'])
                    content = botslib.readdata(row['filename'])     #get attachment from data file
                    if self.userscript and hasattr(self.userscript,'subject'):    #user exit to determine subject
                        subject = botslib.runscript(self.userscript,self.scriptname,'subject',channeldict=self.channeldict,ta=ta_to,subjectstring=subject,content=content)
                    message.add_header('Subject',subject)
@ -305,23 +306,12 @@ class _comsession(object):
                    #set attachment filename
                    filename_mask = self.channeldict['filename'] if self.channeldict['filename'] else '*'
                    attachmentfilename = self.filename_formatter(filename_mask,ta_to)
                    if attachmentfilename and self.channeldict['sendmdn'] != 'body':  #if not explicitly indicated 'as body' or (old)  if attachmentfilename is None or empty string: do not send as an attachment.
                        message.add_header("Content-Disposition",'attachment',filename=attachmentfilename)

                    #set Content-Type and charset
                    charset = self.convertcodecformime(row['charset'])
                    message.add_header('Content-Type',row['contenttype'].lower(),charset=charset)          #contenttype is set in grammar.syntax

                    #set attachment/payload; the Content-Transfer-Encoding is set by python encoder
                    message.set_payload(content)   #do not use charset; this lead to unwanted encodings...bots always uses base64
@ -334,11 +324,8 @@ class _comsession(object):

                    #*******write email to file***************************
                    outfilename = unicode(ta_to.idta)
                    outfile = botslib.opendata(outfilename, 'wb')
                    generator = email.Generator.Generator(outfile, mangle_from_=False, maxheaderlen=78)
                    generator.flatten(message,unixfrom=False)
                    outfile.close()
            except:
@ -361,7 +348,7 @@ class _comsession(object):
            -   filter emails/attachments based on contenttype
            -   email-address should be know by bots (can be turned off)
        '''
        whitelist_multipart = ['multipart/mixed','multipart/digest','multipart/signed','multipart/report','message/rfc822','multipart/alternative']
        whitelist_major = ['text','application']
        blacklist_contenttype = ['text/html','text/enriched','text/rtf','text/richtext','application/postscript','text/vcard','text/css']
        def savemime(msg):
@ -392,12 +379,13 @@ class _comsession(object):
                filesize = len(content)
                ta_file = ta_from.copyta(status=FILEIN)
                outfilename = unicode(ta_file.idta)
                outfile = botslib.opendata(outfilename, 'wb')
                outfile.write(content)
                outfile.close()
                nrmimesaved += 1
                ta_file.update(statust=OK,
                                contenttype=contenttype,
                                charset=charset,
                                filename=outfilename,
                                filesize=filesize)
            return nrmimesaved
@ -405,15 +393,15 @@ class _comsession(object):
        @botslib.log_session
        def mdnreceive():
            tmp = msg.get_param('reporttype')
            if tmp is None or email.Utils.collapse_rfc2231_value(tmp)!='disposition-notification':    #invalid MDN
                raise botslib.CommunicationInError(_(u'Received email-MDN with errors.'))
            for part in msg.get_payload():
                if part.get_content_type()=='message/disposition-notification':
                    originalmessageid = part['original-message-id']
                    if originalmessageid is not None:
                        break
            else:   #invalid MDN: 'message/disposition-notification' not in email
                raise botslib.CommunicationInError(_(u'Received email-MDN with errors.'))
            botslib.changeq('''UPDATE ta
                               SET confirmed=%(confirmed)s, confirmidta=%(confirmidta)s
                               WHERE reference=%(reference)s
@ -430,9 +418,9 @@ class _comsession(object):
                                                            frompartner=frompartner,topartner=topartner):
                return 0 #do not send
            #make message
            message = email.Message.Message()
            message.add_header('From',tomail)
            dispositionnotificationto = email.Utils.parseaddr(msg['disposition-notification-to'])[1]
            message.add_header('To', dispositionnotificationto)
            message.add_header('Subject', 'Return Receipt (displayed) - '+subject)
            message.add_header('MIME-Version','1.0')
@ -441,16 +429,16 @@ class _comsession(object):
            #~ message.set_param('reporttype','disposition-notification')

            #make human readable message
            humanmessage = email.Message.Message()
            humanmessage.add_header('Content-Type', 'text/plain')
            humanmessage.set_payload('This is an return receipt for the mail that you send to '+tomail)
            message.attach(humanmessage)

            #make machine readable message
            machinemessage = email.Message.Message()
            machinemessage.add_header('Content-Type', 'message/disposition-notification')
            machinemessage.add_header('Original-Message-ID', reference)
            nep = email.Message.Message()
            machinemessage.attach(nep)
            message.attach(machinemessage)

@ -459,19 +447,16 @@ class _comsession(object):
            
            if botsglobal.ini.getboolean('acceptance','runacceptancetest',False):
                mdn_reference = '123message-ID email should be unique123'
                mdn_datetime = email.Utils.formatdate(timeval=time.mktime(time.strptime("2013-01-23 01:23:45", "%Y-%m-%d %H:%M:%S")),localtime=True)
            else:
                mdn_reference = email.Utils.make_msgid(unicode(ta_mdn.idta))    #we first have to get the mda-ta to make this reference
                mdn_datetime = email.Utils.formatdate(localtime=True)
            message.add_header('Date',mdn_datetime)
            message.add_header('Message-ID', mdn_reference)
            
            mdnfilename = unicode(ta_mdn.idta)
            mdnfile = botslib.opendata(mdnfilename, 'wb')
            generator = email.Generator.Generator(mdnfile, mangle_from_=False, maxheaderlen=78)
            generator.flatten(message,unixfrom=False)
            mdnfile.close()
            ta_mdn.update(statust=OK,
@ -506,56 +491,51 @@ class _comsession(object):
                confirmasked = False
                confirmidta = 0
                #read & parse email
                ta_from = botslib.OldTransaction(row['idta'])
                infile = botslib.opendata(row['filename'], 'rb')
                msg             = email.message_from_file(infile)   #read and parse mail
                infile.close()
                #******get information from email (sender, receiver etc)***********************************************************
                reference       = self.checkheaderforcharset(msg['message-id'])
                subject = self.checkheaderforcharset(msg['subject'])
                contenttype     = self.checkheaderforcharset(msg.get_content_type())
                #frompartner (incl autorization)
                frommail        = self.checkheaderforcharset(email.Utils.parseaddr(msg['from'])[1])
                frompartner = ''
                if not self.channeldict['starttls']:    #starttls in channeldict is: 'no check on "from:" email adress'
                    frompartner = self.mailaddress2idpartner(frommail)
                    if frompartner is None:
                        raise botslib.CommunicationInError(_(u'"From" emailaddress(es) %(email)s not authorised/unknown for channel "%(idchannel)s".'),
                                                            {'email':frommail,'idchannel':self.channeldict['idchannel']})
                #topartner, cc (incl autorization)
                list_to_address = [self.checkheaderforcharset(address) for name_not_used_variable,address in email.Utils.getaddresses(msg.get_all('to', []))] 
                list_cc_address = [self.checkheaderforcharset(address) for name_not_used_variable,address in email.Utils.getaddresses(msg.get_all('cc', []))] 
                cc_content      = ','.join([address for address in (list_to_address + list_cc_address)])
                topartner = ''  #initialise topartner
                tomail = ''     #initialise tomail
                if not self.channeldict['apop']:    #apop in channeldict is: 'no check on "to:" email adress'
                    for address in list_to_address:   #all tos-addresses are checked; only one needs to be authorised.
                        topartner =  self.mailaddress2idpartner(address)
                        tomail = address
                        if topartner is not None:   #if topartner found: break out of loop
                            break
                    else:   #if no valid topartner: generate error
                        raise botslib.CommunicationInError(_(u'"To" emailaddress(es) %(email)s not authorised/unknown for channel "%(idchannel)s".'),
                                                            {'email':list_to_address,'idchannel':self.channeldict['idchannel']})
 
                #update transaction of mail with information found in mail
                ta_from.update(frommail=frommail,   #why save now not later: when saving the attachments need the amil-header-info to be in ta (copyta)
                                tomail=tomail,
                                reference=reference,
                                contenttype=contenttype,
                                frompartner=frompartner,
                                topartner=topartner,
                                cc = cc_content,
                                rsrv1 = subject)
                if contenttype == 'multipart/report':   #process received MDN confirmation
                    mdnreceive()
                else:
                    if msg.has_key('disposition-notification-To'):  #sender requests a MDN
                        confirmidta = mdnsend(ta_from)
                        if confirmidta:
                            confirmtype = 'send-email-MDN'
@ -563,7 +543,7 @@ class _comsession(object):
                            confirmasked = True
                    nrmimesaved = savemime(msg)
                    if not nrmimesaved:
                        raise botslib.CommunicationInError (_(u'No valid attachment in received email'))
            except:
                txt = botslib.txtexc()
                ta_from.update(statust=ERROR,errortext=txt)
@ -584,13 +564,13 @@ class _comsession(object):
            header.encode('utf8')
            return header
        except:
            raise botslib.CommunicationInError(_(u'Email header invalid - probably issues with characterset.'))
                        
    def mailaddress2idpartner(self,mailaddress):
        ''' lookup email address to see if know in configuration. '''
        mailaddress_lower = mailaddress.lower()
        #first check in chanpar email-addresses for this channel
        for row in botslib.query(u'''SELECT chanpar.idpartner_id as idpartner
                                    FROM chanpar,channel,partner
                                    WHERE chanpar.idchannel_id=channel.idchannel
                                    AND chanpar.idpartner_id=partner.idpartner
@ -598,19 +578,19 @@ class _comsession(object):
                                    AND chanpar.idchannel_id=%(idchannel)s
                                    AND LOWER(chanpar.mail)=%(mail)s''',
                                    {'active':True,'idchannel':self.channeldict['idchannel'],'mail':mailaddress_lower}):
            return row['idpartner']
        #if not found, check in partner-tabel (is less specific). Also test if in CC field.
        for row in botslib.query(u'''SELECT idpartner
                                    FROM partner
                                    WHERE active=%(active)s
                                    AND ( LOWER(mail) = %(mail)s OR LOWER(cc) LIKE %(maillike)s )''',
                                    {'active':True,'mail':mailaddress_lower,'maillike': '%' + mailaddress_lower + '%'}):
            return row['idpartner']
        return None     #indicate email address is unknown


    def idpartner2mailaddress(self,idpartner):
        for row in botslib.query(u'''SELECT chanpar.mail as mail,chanpar.cc as cc
                                    FROM chanpar,channel,partner
                                    WHERE chanpar.idchannel_id=channel.idchannel
                                    AND chanpar.idpartner_id=partner.idpartner
@ -618,16 +598,16 @@ class _comsession(object):
                                    AND chanpar.idchannel_id=%(idchannel)s
                                    AND chanpar.idpartner_id=%(idpartner)s''',
                                    {'active':True,'idchannel':self.channeldict['idchannel'],'idpartner':idpartner}):
            if row['mail']:
                return row['mail'],row['cc']
        for row in botslib.query(u'''SELECT mail,cc
                                    FROM partner
                                    WHERE active=%(active)s
                                    AND idpartner=%(idpartner)s''',
                                    {'active':True,'idpartner':idpartner}):
            if row['mail']:
                return row['mail'],row['cc']
        raise botslib.CommunicationOutError(_(u'No mail-address for partner "%(partner)s" (channel "%(idchannel)s").'),
                                                {'partner':idpartner,'idchannel':self.channeldict['idchannel']})

    def connect(self):
@ -649,13 +629,13 @@ class _comsession(object):


    def filename_formatter(self,filename_mask,ta):
        ''' Output filename generation from "template" filename configured in the channel
            Basically python's string.Formatter is used; see http://docs.python.org/library/string.html
            As in string.Formatter, substitution values are surrounded by braces; format specifiers can be used.
            Any ta value can be used
              eg. {botskey}, {alt}, {editype}, {messagetype}, {topartner}
            Next to the value in ta you can use:
            -   * : an unique number (per outchannel) using an asterisk
            -   {datetime}  use datetime with a valid strftime format: 
                eg. {datetime:%Y%m%d}, {datetime:%H%M%S}
            -   {infile} use the original incoming filename; use name and extension, or either part separately:
@ -663,19 +643,19 @@ class _comsession(object):
            -   {overwrite}  if file wit hfielname exists: overwrite it (instead of appending)
            
            Exampels of usage:
                {botskey}_*.idoc        use incoming order number, add unique number, use extension '.idoc'
                *_{infile}              passthrough incoming filename & extension, prepend with unique number
                {infile:name}_*.txt     passthrough incoming filename, add unique number but change extension to .txt
                {editype}-{messagetype}-{datetime:%Y%m%d}-*.{infile:ext}
                                        use editype, messagetype, date and unique number with extension from the incoming file
                {topartner}/{messagetype}/*.edi
                                        Usage of subdirectories in the filename, they must already exist. In the example:
                                        sort into folders by partner and messagetype.
                        
            Note1: {botskey} can only be used if merge is False for that messagetype
        '''
        class infilestr(str):
            ''' class for the infile-string that handles the specific format-options'''
            def __format__(self, format_spec):
                if not format_spec:
                    return unicode(self)
@ -686,31 +666,30 @@ class _comsession(object):
                    return ext 
                if format_spec == 'name':
                    return name 
                raise botslib.CommunicationOutError(_(u'Error in format of "{filename}": unknown format: "%(format)s".'),
                                                    {'format':format_spec})
        unique = unicode(botslib.unique(self.channeldict['idchannel'])) #create unique part for attachment-filename
        tofilename = filename_mask.replace('*',unique)           #filename_mask is filename in channel where '*' is replaced by idta
        if '{' in tofilename:    #only for python 2.6/7
            ta.synall()
            if '{infile' in tofilename:
                ta_list = botslib.trace_origin(ta=ta,where={'status':EXTERNIN})
                if ta_list:
                    infilename = infilestr(os.path.basename(ta_list[-1].filename))
                else:
                    infilename = ''
            else:
                infilename = ''
            try:
                if botsglobal.ini.getboolean('acceptance','runacceptancetest',False):
                    datetime_object = datetime.datetime.strptime("2013-01-23 01:23:45", "%Y-%m-%d %H:%M:%S")
                else:
                    datetime_object = datetime.datetime.now()
                tofilename = tofilename.format(infile=infilename,datetime=datetime_object,**ta.__dict__)
            except:
                txt = botslib.txtexc()
                raise botslib.CommunicationOutError(_(u'Error in formatting outgoing filename "%(filename)s". Error: "%(error)s".'),
                                                        {'filename':tofilename,'error':txt})
        if self.userscript and hasattr(self.userscript,'filename'):
            return botslib.runscript(self.userscript,self.scriptname,'filename',channeldict=self.channeldict,filename=tofilename,ta=ta)
        else:
@ -729,7 +708,8 @@ class file(_comsession):
        ''' gets files from filesystem.
        '''
        frompath = botslib.join(self.channeldict['path'],self.channeldict['filename'])
        filelist = [filename for filename in glob.iglob(frompath) if os.path.isfile(filename)]
        filelist.sort()
        startdatetime = datetime.datetime.now()
        remove_ta = False
        for fromfilename in filelist:
@ -749,10 +729,10 @@ class file(_comsession):
                    elif os.name == 'posix':
                        fcntl.lockf(fromfile.fileno(), fcntl.LOCK_SH|fcntl.LOCK_NB)
                    else:
                        raise botslib.LockedFileError(_(u'Can not do a systemlock on this platform'))
                #open tofile
                tofilename = unicode(ta_to.idta)
                tofile = botslib.opendata(tofilename, 'wb')
                #copy
                shutil.copyfileobj(fromfile,tofile,1048576)
                tofile.close()
@ -794,7 +774,7 @@ class file(_comsession):
        else:
            mode = 'ab'
        #select the db-ta's for this channel
        for row in botslib.query(u'''SELECT idta,filename,numberofresends
                                       FROM ta
                                      WHERE idta>%(rootidta)s
                                        AND status=%(status)s
@ -805,7 +785,7 @@ class file(_comsession):
                                    {'tochannel':self.channeldict['idchannel'],'rootidta':self.rootidta,
                                    'status':FILEOUT,'statust':OK}):
            try:    #for each db-ta:
                ta_from = botslib.OldTransaction(row['idta'])
                ta_to =   ta_from.copyta(status=EXTERNOUT)
                #open tofile, incl syslock if indicated
                tofilename = self.filename_formatter(filename_mask,ta_from)
@ -817,9 +797,9 @@ class file(_comsession):
                    elif os.name == 'posix':
                        fcntl.lockf(tofile.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
                    else:
                        raise botslib.LockedFileError(_(u'Can not do a systemlock on this platform'))
                #open fromfile
                fromfile = botslib.opendata(row['filename'], 'rb')
                #copy
                shutil.copyfileobj(fromfile,tofile,1048576)
                fromfile.close()
@ -833,9 +813,9 @@ class file(_comsession):
                    os.rename(tofilename_old,tofilename)
            except:
                txt = botslib.txtexc()
                ta_to.update(statust=ERROR,errortext=txt,numberofresends=row['numberofresends']+1)
            else:
                ta_to.update(statust=DONE,filename=tofilename,numberofresends=row['numberofresends']+1)
            finally:
                ta_from.update(statust=DONE)

@ -875,9 +855,9 @@ class pop3(_comsession):
                tofilename = unicode(ta_to.idta)
                mailid = int(mail.split()[0])  #first 'word' is the message number/ID
                maillines = self.session.retr(mailid)[1]        #alt: (header, messagelines, octets) = popsession.retr(messageID)
                tofile = botslib.opendata(tofilename, 'wb')
                content = os.linesep.join(maillines)
                filesize = len(content)
                tofile.write(content)
                tofile.close()
                if self.channeldict['remove']:      #on server side mail is marked to be deleted. The pop3-server will actually delete the file if the QUIT commnd is receieved!
@ -911,10 +891,10 @@ class pop3(_comsession):
    def disconnect(self):
        try:
            if not self.session:
                raise Exception(_(u'Pop3 connection not OK'))
            resp = self.session.quit()     #pop3 server will now actually delete the mails
            if resp[:1] != '+':
                raise Exception(_(u'QUIT command to POP3 server failed'))
        except Exception:   #connection is gone. Delete everything that is received to avoid double receiving.
            botslib.ErrorProcess(functionname='pop3-incommunicate',errortext='Could not fetch emails via POP3; probably communication problems',channeldict=self.channeldict)
            for idta in self.listoftamarkedfordelete:
@ -928,7 +908,14 @@ class pop3(_comsession):
class pop3s(pop3):
    def connect(self):
        import poplib
        if self.userscript and hasattr(self.userscript,'keyfile'):
            keyfile, certfile = botslib.runscript(self.userscript,self.scriptname,'keyfile',channeldict=self.channeldict)
        elif self.channeldict['keyfile']:
            keyfile = self.channeldict['keyfile']
            certfile = self.channeldict['certfile']
        else:
            keyfile = certfile = None
        self.session = poplib.POP3_SSL(host=self.channeldict['host'],port=int(self.channeldict['port']),keyfile=keyfile,certfile=certfile)
        self.session.set_debuglevel(botsglobal.ini.getint('settings','pop3debug',0))    #if used, gives information about session (on screen), for debugging pop3
        self.session.user(self.channeldict['username'])
        self.session.pass_(self.channeldict['secret'])
@ -982,7 +969,7 @@ class imap4(_comsession):
                filename = unicode(ta_to.idta)
                # Get the message (header and body)
                response, msg_data = self.session.uid('fetch',mail, '(RFC822)')
                filehandler = botslib.opendata(filename, 'wb')
                filesize = len(msg_data[0][1])
                filehandler.write(msg_data[0][1])
                filehandler.close()
@ -1019,8 +1006,15 @@ class imap4(_comsession):
class imap4s(imap4):
    def connect(self):
        import imaplib
        if self.userscript and hasattr(self.userscript,'keyfile'):
            keyfile, certfile = botslib.runscript(self.userscript,self.scriptname,'keyfile',channeldict=self.channeldict)
        elif self.channeldict['keyfile']:
            keyfile = self.channeldict['keyfile']
            certfile = self.channeldict['certfile']
        else:
            keyfile = certfile = None
        imaplib.Debug = botsglobal.ini.getint('settings','imap4debug',0)    #if used, gives information about session (on screen), for debugging imap4
        self.session = imaplib.IMAP4_SSL(host=self.channeldict['host'],port=int(self.channeldict['port']),keyfile=keyfile,certfile=certfile)
        self.session.login(self.channeldict['username'],self.channeldict['secret'])


@ -1040,10 +1034,10 @@ class smtp(_comsession):
                #error in python 2.6.4....user and password can not be unicode
                self.session.login(str(self.channeldict['username']),str(self.channeldict['secret']))
            except smtplib.SMTPAuthenticationError:
                raise botslib.CommunicationOutError(_(u'SMTP server did not accept user/password combination.'))
            except:
                txt = botslib.txtexc()
                raise botslib.CommunicationOutError(_(u'SMTP login failed. Error:\n%(txt)s'),{'txt':txt})

    @botslib.log_session
    def outcommunicate(self):
@ -1051,7 +1045,7 @@ class smtp(_comsession):
            SMTP does not allow rollback. So if the sending of a mail fails, other mails may have been send.
        '''
        #send messages
        for row in botslib.query(u'''SELECT idta,filename,frommail,tomail,cc,numberofresends
                                    FROM ta
                                    WHERE idta>%(rootidta)s
                                    AND status=%(status)s
@ -1061,19 +1055,19 @@ class smtp(_comsession):
                                    {'status':FILEOUT,'statust':OK,'rootidta':self.rootidta,
                                    'tochannel':self.channeldict['idchannel']}):
            try:
                ta_from = botslib.OldTransaction(row['idta'])
                ta_to = ta_from.copyta(status=EXTERNOUT)
                addresslist = row['tomail'].split(',') + row['cc'].split(',')
                addresslist = [x.strip() for x in addresslist if x.strip()]
                sendfile = botslib.opendata(row['filename'], 'rb')
                msg = sendfile.read()
                sendfile.close()
                self.session.sendmail(row['frommail'], addresslist, msg)
            except:
                txt = botslib.txtexc()
                ta_to.update(statust=ERROR,errortext=txt,filename='smtp://'+self.channeldict['username']+'@'+self.channeldict['host'],numberofresends=row['numberofresends']+1)
            else:
                ta_to.update(statust=DONE,filename='smtp://'+self.channeldict['username']+'@'+self.channeldict['host'],numberofresends=row['numberofresends']+1)
            finally:
                ta_from.update(statust=DONE)

@ -1087,16 +1081,30 @@ class smtp(_comsession):

class smtps(smtp):
    def connect(self):
        if self.userscript and hasattr(self.userscript,'keyfile'):
            keyfile, certfile = botslib.runscript(self.userscript,self.scriptname,'keyfile',channeldict=self.channeldict)
        elif self.channeldict['keyfile']:
            keyfile = self.channeldict['keyfile']
            certfile = self.channeldict['certfile']
        else:
            keyfile = certfile = None
        self.session = smtplib.SMTP_SSL(host=self.channeldict['host'],port=int(self.channeldict['port']),keyfile=keyfile,certfile=certfile) #make connection
        self.session.set_debuglevel(botsglobal.ini.getint('settings','smtpdebug',0))    #if used, gives information about session (on screen), for debugging smtp
        self.login()

class smtpstarttls(smtp):
    def connect(self):
        if self.userscript and hasattr(self.userscript,'keyfile'):
            keyfile, certfile = botslib.runscript(self.userscript,self.scriptname,'keyfile',channeldict=self.channeldict)
        elif self.channeldict['keyfile']:
            keyfile = self.channeldict['keyfile']
            certfile = self.channeldict['certfile']
        else:
            keyfile = certfile = None
        self.session = smtplib.SMTP(host=self.channeldict['host'],port=int(self.channeldict['port'])) #make connection
        self.session.set_debuglevel(botsglobal.ini.getint('settings','smtpdebug',0))    #if used, gives information about session (on screen), for debugging smtp
        self.session.ehlo()
        self.session.starttls(keyfile=keyfile,certfile=certfile)
        self.session.ehlo()
        self.login()

@ -1135,16 +1143,12 @@ class ftp(_comsession):
            each to be imported file is transaction.
            each imported file is transaction.
        '''
        startdatetime = datetime.datetime.now()
        files = []
        try:            #some ftp servers give errors when directory is empty; catch these errors here
            files = self.session.nlst()
        except (ftplib.error_perm,ftplib.error_temp) as msg:
            if unicode(msg)[:3] not in [u'550',u'450']:
                raise

        lijst = fnmatch.filter(files,self.channeldict['filename'])
@ -1158,22 +1162,21 @@ class ftp(_comsession):
                ta_to =   ta_from.copyta(status=FILEIN)
                remove_ta = True
                tofilename = unicode(ta_to.idta)
                tofile = botslib.opendata(tofilename, 'wb')
                try:
                    if self.channeldict['ftpbinary']:
                        self.session.retrbinary("RETR " + fromfilename, tofile.write)
                    else:
                        self.session.retrlines("RETR " + fromfilename, lambda s, w=tofile.write: w(s+"\n"))
                except ftplib.error_perm as msg:
                    if unicode(msg)[:3] in [u'550',]:     #we are trying to download a directory...
                        raise botslib.BotsError(u'To be catched')
                    else:
                        raise
                tofile.close()
                filesize = os.path.getsize(botslib.abspathdata(tofilename))
                if not filesize:
                    raise botslib.BotsError(u'To be catched; directory (or empty file)')
            except botslib.BotsError:   #directory or empty file; handle exception but generate no error.
                if remove_ta:
                    try:
@ -1224,13 +1227,14 @@ class ftp(_comsession):
                                    {'tochannel':self.channeldict['idchannel'],'rootidta':self.rootidta,
                                    'status':FILEOUT,'statust':OK}):
            try:
                ta_from = botslib.OldTransaction(row['idta'])
                ta_to = ta_from.copyta(status=EXTERNOUT)
                tofilename = self.filename_formatter(filename_mask,ta_from)
                if self.channeldict['ftpbinary']:
                    fromfile = botslib.opendata(row['filename'], 'rb')
                    self.session.storbinary(mode + tofilename, fromfile)
                else:
                    fromfile = botslib.opendata(row['filename'], 'r')
                    self.session.storlines(mode + tofilename, fromfile)
                fromfile.close()
                #Rename filename after writing file.
@ -1241,9 +1245,9 @@ class ftp(_comsession):
                    self.session.rename(tofilename_old,tofilename)
            except:
                txt = botslib.txtexc()
                ta_to.update(statust=ERROR,errortext=txt,filename='ftp:/'+posixpath.join(self.dirpath,tofilename),numberofresends=row['numberofresends']+1)
            else:
                ta_to.update(statust=DONE,filename='ftp:/'+posixpath.join(self.dirpath,tofilename),numberofresends=row['numberofresends']+1)
            finally:
                ta_from.update(statust=DONE)

@ -1261,9 +1265,16 @@ class ftps(ftp):
    '''
    def connect(self):
        if not hasattr(ftplib,'FTP_TLS'):
            raise botslib.CommunicationError(_(u'ftps is not supported by your python version, use >=2.7'))
        if self.userscript and hasattr(self.userscript,'keyfile'):
            keyfile, certfile = botslib.runscript(self.userscript,self.scriptname,'keyfile',channeldict=self.channeldict)
        elif self.channeldict['keyfile']:
            keyfile = self.channeldict['keyfile']
            certfile = self.channeldict['certfile']
        else:
            keyfile = certfile = None
        botslib.settimeout(botsglobal.ini.getint('settings','ftptimeout',10))
        self.session = ftplib.FTP_TLS(keyfile=keyfile,certfile=certfile)
        self.session.set_debuglevel(botsglobal.ini.getint('settings','ftpdebug',0))   #set debug level (0=no, 1=medium, 2=full debug)
        self.session.set_pasv(not self.channeldict['ftpactive']) #active or passive ftp
        self.session.connect(host=self.channeldict['host'],port=int(self.channeldict['port']))
@ -1294,10 +1305,7 @@ if hasattr(ftplib,'FTP_TLS'):
            #added hje 20110713: directly use SSL in FTPIS
            self.sock = ssl.wrap_socket(self.sock, self.keyfile, self.certfile,ssl_version=self.ssl_version)
            #end added
            self.file = self.sock.makefile('rb')
            self.welcome = self.getresp()
            return self.welcome
        def prot_p(self):
@ -1332,9 +1340,16 @@ class ftpis(ftp):
    '''
    def connect(self):
        if not hasattr(ftplib,'FTP_TLS'):
            raise botslib.CommunicationError(_(u'ftpis is not supported by your python version, use >=2.7'))
        if self.userscript and hasattr(self.userscript,'keyfile'):
            keyfile, certfile = botslib.runscript(self.userscript,self.scriptname,'keyfile',channeldict=self.channeldict)
        elif self.channeldict['keyfile']:
            keyfile = self.channeldict['keyfile']
            certfile = self.channeldict['certfile']
        else:
            keyfile = certfile = None
        botslib.settimeout(botsglobal.ini.getint('settings','ftptimeout',10))
        self.session = Ftp_tls_implicit(keyfile=keyfile,certfile=certfile)
        if self.channeldict['parameters']:
            self.session.ssl_version = int(self.channeldict['parameters'])
        self.session.set_debuglevel(botsglobal.ini.getint('settings','ftpdebug',0))   #set debug level (0=no, 1=medium, 2=full debug)
@ -1361,11 +1376,11 @@ class sftp(_comsession):
        try:
            import paramiko
        except:
            raise ImportError(_(u'Dependency failure: communicationtype "sftp" requires python library "paramiko".'))
        try:
            from Crypto import Cipher
        except:
            raise ImportError(_(u'Dependency failure: communicationtype "sftp" requires python library "pycrypto".'))
        # setup logging if required
        ftpdebug = botsglobal.ini.getint('settings','ftpdebug',0)
        if ftpdebug > 0:
@ -1440,10 +1455,10 @@ class sftp(_comsession):
                ta_to =   ta_from.copyta(status=FILEIN)
                remove_ta = True
                tofilename = unicode(ta_to.idta)
                fromfile = self.session.open(fromfilename, 'r')    # SSH treats all files as binary
                content = fromfile.read()
                filesize = len(content)
                tofile = botslib.opendata(tofilename, 'wb')
                tofile.write(content)
                tofile.close()
                fromfile.close()
@ -1489,11 +1504,11 @@ class sftp(_comsession):
                                    {'tochannel':self.channeldict['idchannel'],'rootidta':self.rootidta,
                                    'status':FILEOUT,'statust':OK}):
            try:
                ta_from = botslib.OldTransaction(row['idta'])
                ta_to = ta_from.copyta(status=EXTERNOUT)
                tofilename = self.filename_formatter(filename_mask,ta_from)
                fromfile = botslib.opendata(row['filename'], 'rb')
                tofile = self.session.open(tofilename, mode)    # SSH treats all files as binary
                tofile.write(fromfile.read())
                tofile.close()
                fromfile.close()
@ -1505,9 +1520,9 @@ class sftp(_comsession):
                    self.session.rename(tofilename_old,tofilename)
            except:
                txt = botslib.txtexc()
                ta_to.update(statust=ERROR,errortext=txt,filename='sftp:/'+posixpath.join(self.dirpath,tofilename),numberofresends=row['numberofresends']+1)
            else:
                ta_to.update(statust=DONE,filename='sftp:/'+posixpath.join(self.dirpath,tofilename),numberofresends=row['numberofresends']+1)
            finally:
                ta_from.update(statust=DONE)

@ -1518,12 +1533,9 @@ class xmlrpc(_comsession):
        From channel is used: usernaem, secret, host, port, path. Path is the function to be used/called.
    '''
    def connect(self):
        import xmlrpclib
        uri = "http://%(username)s%(secret)s@%(host)s:%(port)s"%self.channeldict
        self.filename = "http://%(username)s@%(host)s:%(port)s"%self.channeldict    #used as 'filename' in reports etc
        session = xmlrpclib.ServerProxy(uri)
        self.xmlrpc_call = getattr(session,self.channeldict['path'])                #self.xmlrpc_call is called in communication

@ -1543,7 +1555,7 @@ class xmlrpc(_comsession):
                ta_to =   ta_from.copyta(status=FILEIN)
                remove_ta = True
                tofilename = unicode(ta_to.idta)
                tofile = botslib.opendata(tofilename, 'wb')
                simplejson.dump(content, tofile, skipkeys=False, ensure_ascii=False, check_circular=False)
                tofile.close()
                filesize = os.path.getsize(botslib.abspathdata(tofilename))
@ -1580,17 +1592,17 @@ class xmlrpc(_comsession):
                                    {'tochannel':self.channeldict['idchannel'],'rootidta':self.rootidta,
                                    'status':FILEOUT,'statust':OK}):
            try:
                ta_from = botslib.OldTransaction(row['idta'])
                ta_to =   ta_from.copyta(status=EXTERNOUT)
                fromfile = botslib.opendata(row['filename'], 'rb',row['charset'])
                content = fromfile.read()
                fromfile.close()
                response = self.xmlrpc_call(content)
            except:
                txt = botslib.txtexc()
                ta_to.update(statust=ERROR,errortext=txt,numberofresends=row['numberofresends']+1)
            else:
                ta_to.update(statust=DONE,filename=self.filename,numberofresends=row['numberofresends']+1)
            finally:
                ta_from.update(statust=DONE)

@ -1608,20 +1620,20 @@ class db(_comsession):
    '''
    def connect(self):
        if self.userscript is None:
            raise botslib.BotsImportError(_(u'Channel "%(idchannel)s" is type "db", but no communicationscript exists.'),
                                {'idchannel':self.channeldict['idchannel']})
        #check functions bots assumes to be present in userscript:
        if not hasattr(self.userscript,'connect'):
            raise botslib.ScriptImportError(_(u'No function "connect" in imported communicationscript "%(communicationscript)s".'),
                                                {'communicationscript':self.scriptname})
        if self.channeldict['inorout'] == 'in' and not hasattr(self.userscript,'incommunicate'):
            raise botslib.ScriptImportError(_(u'No function "incommunicate" in imported communicationscript "%(communicationscript)s".'),
                                                {'communicationscript':self.scriptname})
        if self.channeldict['inorout'] == 'out' and not hasattr(self.userscript,'outcommunicate'):
            raise botslib.ScriptImportError(_(u'No function "outcommunicate" in imported communicationscript "%(communicationscript)s".'),
                                                {'communicationscript':self.scriptname})
        if not hasattr(self.userscript,'disconnect'):
            raise botslib.ScriptImportError(_(u'No function "disconnect" in imported communicationscript "%(communicationscript)s".'),
                                            {'communicationscript':self.scriptname})

        self.dbconnection = botslib.runscript(self.userscript,self.scriptname,'connect',channeldict=self.channeldict)
@ -1629,12 +1641,12 @@ class db(_comsession):
    @botslib.log_session
    def incommunicate(self):
        ''' read data from database.
            userscript should return a 'db_objects'.
            This can be one edi-message or several edi-messages.
            if a list or tuple is passed: each element of list/tuple is treated as seperate edi-message.
            if this is None, do nothing
            if this is a list/tuple, each member of the list is send as a separate 'message'
            if you want all information from userscript to be passed as one edi message: pass as dict, eg {'data': <list of queries>}
        '''
        db_objects = botslib.runscript(self.userscript,self.scriptname,'incommunicate',channeldict=self.channeldict,dbconnection=self.dbconnection)
        if not db_objects:      #there should be a useful db_objects; if not just return (do nothing)
@ -1652,7 +1664,9 @@ class db(_comsession):
                ta_to = ta_from.copyta(status=FILEIN)
                remove_ta = True
                tofilename = unicode(ta_to.idta)
                tofile = botslib.opendata(tofilename,'wb')
                pickle.dump(db_object, tofile)
                tofile.close()
                filesize = os.path.getsize(botslib.abspathdata(tofilename))
            except:
                txt = botslib.txtexc()
@ -1671,7 +1685,7 @@ class db(_comsession):

    @botslib.log_session
    def outcommunicate(self):
        ''' write data to database.
        '''
        for row in botslib.query('''SELECT idta,filename,numberofresends
                                    FROM ta
@ -1681,15 +1695,17 @@ class db(_comsession):
                                    AND tochannel=%(tochannel)s ''',
                                    {'tochannel':self.channeldict['idchannel'],'rootidta':self.rootidta,'status':FILEOUT,'statust':OK}):
            try:
                ta_from = botslib.OldTransaction(row['idta'])
                ta_to = ta_from.copyta(status=EXTERNOUT)
                fromfile = botslib.opendata(row['filename'], 'rb')
                db_object = pickle.load(fromfile)
                fromfile.close()
                botslib.runscript(self.userscript,self.scriptname,'outcommunicate',channeldict=self.channeldict,dbconnection=self.dbconnection,db_object=db_object)
            except:
                txt = botslib.txtexc()
                ta_to.update(statust=ERROR,errortext=txt,filename=self.channeldict['path'],numberofresends=row['numberofresends']+1)
            else:
                ta_to.update(statust=DONE,filename=self.channeldict['path'],numberofresends=row['numberofresends']+1)
            finally:
                ta_from.update(statust=DONE)

@ -1725,7 +1741,7 @@ class communicationscript(_comsession):
    ''' 
    def connect(self):
        if self.userscript is None or not botslib.tryrunscript(self.userscript,self.scriptname,'connect',channeldict=self.channeldict):
            raise botslib.BotsImportError(_(u'Channel "%(idchannel)s" is type "communicationscript", but no communicationscript exists.') ,
                                {'idchannel':self.channeldict})


@ -1747,7 +1763,7 @@ class communicationscript(_comsession):
                    filesize = os.fstat(fromfile.fileno()).st_size
                    #open tofile
                    tofilename = unicode(ta_to.idta)
                    tofile = botslib.opendata(tofilename, 'wb')
                    #copy
                    shutil.copyfileobj(fromfile,tofile,1048576)
                    fromfile.close()
@ -1772,7 +1788,8 @@ class communicationscript(_comsession):
                        break
        else:   #all files have been set ready by external communicationscript using 'connect'.
            frompath = botslib.join(self.channeldict['path'], self.channeldict['filename'])
            filelist = [filename for filename in glob.iglob(frompath) if os.path.isfile(filename)]
            filelist.sort()
            remove_ta = False
            for fromfilename in filelist:
                try:
@ -1784,7 +1801,7 @@ class communicationscript(_comsession):
                    remove_ta = True
                    fromfile = open(fromfilename, 'rb')
                    tofilename = unicode(ta_to.idta)
                    tofile = botslib.opendata(tofilename, 'wb')
                    content = fromfile.read()
                    filesize = len(content)
                    tofile.write(content)
@ -1823,7 +1840,7 @@ class communicationscript(_comsession):
        else:
            mode = 'ab'
        #select the db-ta's for this channel
        for row in botslib.query(u'''SELECT idta,filename,numberofresends
                                    FROM ta
                                    WHERE idta>%(rootidta)s
                                    AND status=%(status)s
@ -1832,14 +1849,14 @@ class communicationscript(_comsession):
                                    {'tochannel':self.channeldict['idchannel'],'rootidta':self.rootidta,
                                    'status':FILEOUT,'statust':OK}):
            try:    #for each db-ta:
                ta_from = botslib.OldTransaction(row['idta'])
                ta_to =   ta_from.copyta(status=EXTERNOUT)
                tofilename = self.filename_formatter(filename_mask,ta_from)
                #open tofile
                tofilename = botslib.join(outputdir,tofilename)
                tofile = open(tofilename, mode)
                #open fromfile
                fromfile = botslib.opendata(row['filename'], 'rb')
                #copy
                shutil.copyfileobj(fromfile,tofile,1048576)
                fromfile.close()
@ -1850,9 +1867,9 @@ class communicationscript(_comsession):
                        os.remove(tofilename)
            except:
                txt = botslib.txtexc()
                ta_to.update(statust=ERROR,errortext=txt,numberofresends=row['numberofresends']+1)
            else:
                ta_to.update(statust=DONE,filename=tofilename,numberofresends=row['numberofresends']+1)
            finally:
                ta_from.update(statust=DONE)

@ -1873,7 +1890,7 @@ class trash(_comsession):
        ''' does output of files to 'nothing' (trash it).
        '''
        #select the db-ta's for this channel
        for row in botslib.query(u'''SELECT idta,filename,numberofresends
                                       FROM ta
                                      WHERE idta>%(rootidta)s
                                        AND status=%(status)s
@ -1883,13 +1900,13 @@ class trash(_comsession):
                                    {'tochannel':self.channeldict['idchannel'],'rootidta':self.rootidta,
                                    'status':FILEOUT,'statust':OK}):
            try:    #for each db-ta:
                ta_from = botslib.OldTransaction(row['idta'])
                ta_to =   ta_from.copyta(status=EXTERNOUT)
            except:
                txt = botslib.txtexc()
                ta_to.update(statust=ERROR,errortext=txt,numberofresends=row['numberofresends']+1)
            else:
                ta_to.update(statust=DONE,filename='',numberofresends=row['numberofresends']+1)
            finally:
                ta_from.update(statust=DONE)

@ -1903,7 +1920,7 @@ class http(_comsession):
        try:
            self.requests = botslib.botsbaseimport('requests')
        except ImportError:
            raise ImportError(_(u'Dependency failure: communicationtype "http(s)" requires python library "requests".'))
        if self.channeldict['username'] and self.channeldict['secret']:
            self.auth = (self.channeldict['username'], self.channeldict['secret'])
        else:
@ -1925,7 +1942,7 @@ class http(_comsession):
                                                headers=self.headers,
                                                verify=self.verify)
                if outResponse.status_code != self.requests.codes.ok: #communication not OK: exception
                    raise botslib.CommunicationError(_(u'%(scheme)s receive error, response code: "%(status_code)s".'),{'scheme':self.scheme,'status_code':outResponse.status_code})
                if not outResponse.content: #communication OK, but nothing received: break
                    break
                ta_from = botslib.NewTransaction(filename=self.url.uri(),
@ -1935,7 +1952,7 @@ class http(_comsession):
                ta_to =   ta_from.copyta(status=FILEIN)
                remove_ta = True
                tofilename = unicode(ta_to.idta)
                tofile = botslib.opendata(tofilename, 'wb')
                tofile.write(outResponse.content)
                tofile.close()
                filesize = len(outResponse.content)
@ -1976,9 +1993,9 @@ class http(_comsession):
                                    {'tochannel':self.channeldict['idchannel'],'rootidta':self.rootidta,
                                    'status':FILEOUT,'statust':OK}):
            try:
                ta_from = botslib.OldTransaction(row['idta'])
                ta_to = ta_from.copyta(status=EXTERNOUT)
                fromfile = botslib.opendata(row['filename'], 'rb')
                content = fromfile.read()
                fromfile.close()
                #communicate via requests library
@ -1990,12 +2007,12 @@ class http(_comsession):
                                                data=content,
                                                verify=self.verify)
                if outResponse.status_code != self.requests.codes.ok:
                    raise botslib.CommunicationError(_(u'%(scheme)s send error, response code: "%(status_code)s".'),{'scheme':self.scheme,'status_code':outResponse.status_code})
            except:
                txt = botslib.txtexc()
                ta_to.update(statust=ERROR,errortext=txt,filename=self.url.uri(filename=row['filename']),numberofresends=row['numberofresends']+1)
            else:
                ta_to.update(statust=DONE,filename=self.url.uri(filename=row['filename']),numberofresends=row['numberofresends']+1)
            finally:
                ta_from.update(statust=DONE)

@ -2017,7 +2034,7 @@ class https(http):
        #option to set environement variable for requests library; use if https server has an unrecognized CA
        super(https,self).connect()
        if self.caCert:
            os.environ["REQUESTS_CA_BUNDLE"] = self.caCert
        if self.channeldict['certfile'] and self.channeldict['keyfile']:
            self.cert = (self.channeldict['certfile'], self.channeldict['keyfile'])
        else:
 ⏳ tiktoken

tiktoken is a fast [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokeniser for use with
OpenAI's models.

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4o")
```

The open source version of `tiktoken` can be installed from PyPI:
```
pip install tiktoken
```

The tokeniser API is documented in `tiktoken/core.py`.

Example code using `tiktoken` can be found in the
[OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb).


## Performance

`tiktoken` is between 3-6x faster than a comparable open source tokeniser:

![image](https://raw.githubusercontent.com/openai/tiktoken/main/perf.svg)

Performance measured on 1GB of text using the GPT-2 tokeniser, using `GPT2TokenizerFast` from
`tokenizers==0.13.2`, `transformers==4.24.0` and `tiktoken==0.2.0`.


## Getting help

Please post questions in the [issue tracker](https://github.com/openai/tiktoken/issues).

If you work at OpenAI, make sure to check the internal documentation or feel free to contact
@shantanu.

## What is BPE anyway?

Language models don't see text like you and I, instead they see a sequence of numbers (known as tokens).
Byte pair encoding (BPE) is a way of converting text into tokens. It has a couple desirable
properties:
1) It's reversible and lossless, so you can convert tokens back into the original text
2) It works on arbitrary text, even text that is not in the tokeniser's training data
3) It compresses the text: the token sequence is shorter than the bytes corresponding to the
   original text. On average, in practice, each token corresponds to about 4 bytes.
4) It attempts to let the model see common subwords. For instance, "ing" is a common subword in
   English, so BPE encodings will often split "encoding" into tokens like "encod" and "ing"
   (instead of e.g. "enc" and "oding"). Because the model will then see the "ing" token again and
   again in different contexts, it helps models generalise and better understand grammar.

`tiktoken` contains an educational submodule that is friendlier if you want to learn more about
the details of BPE, including code that helps visualise the BPE procedure:
```python
from tiktoken._educational import *

# Train a BPE tokeniser on a small amount of text
enc = train_simple_encoding()

# Visualise how the GPT-4 encoder encodes text
enc = SimpleBytePairEncoding.from_tiktoken("cl100k_base")
enc.encode("hello world aaaaaaaaaaaa")
```


## Extending tiktoken

You may wish to extend `tiktoken` to support new encodings. There are two ways to do this.


**Create your `Encoding` object exactly the way you want and simply pass it around.**

```python
cl100k_base = tiktoken.get_encoding("cl100k_base")

# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings
enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    }
)
```

**Use the `tiktoken_ext` plugin mechanism to register your `Encoding` objects with `tiktoken`.**

This is only useful if you need `tiktoken.get_encoding` to find your encoding, otherwise prefer
option 1.

To do this, you'll need to create a namespace package under `tiktoken_ext`.

Layout your project like this, making sure to omit the `tiktoken_ext/__init__.py` file:
```
my_tiktoken_extension
├── tiktoken_ext
│   └── my_encodings.py
└── setup.py
```

`my_encodings.py` should be a module that contains a variable named `ENCODING_CONSTRUCTORS`.
This is a dictionary from an encoding name to a function that takes no arguments and returns
arguments that can be passed to `tiktoken.Encoding` to construct that encoding. For an example, see
`tiktoken_ext/openai_public.py`. For precise details, see `tiktoken/registry.py`.

Your `setup.py` should look something like this:
```python
from setuptools import setup, find_namespace_packages

setup(
    name="my_tiktoken_extension",
    packages=find_namespace_packages(include=['tiktoken_ext*']),
    install_requires=["tiktoken"],
    ...
)
```

Then simply `pip install ./my_tiktoken_extension` and you should be able to use your
custom encodings! Make sure **not** to use an editable install.
